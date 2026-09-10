import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import datetime
import time
import os

# =========================================================================
# SHARED TENSORBOARD TAGS (identical in baseline / DDP / pipeline):
#   Loss/train, Loss/test, Accuracy/train, Accuracy/test   (x = epoch)
#   Accuracy/test_over_training_time                       (x = train seconds)
#   Time/total_training_seconds                            (single value)
# Kubernetes version: launched by torchrun, one process per pod.
# Logging + model save happen on rank 0 only.
#   runs        -> /app/runs        (output-volume, host: .../runs)
#   model.pth/  -> /app/model.pth   (model-volume,  host: .../distributed)
# =========================================================================


def setup():
    """Join the torchrun rendezvous.

    torchrun (one process per pod via the K8s Job) populates RANK, WORLD_SIZE,
    MASTER_ADDR and MASTER_PORT in the environment through the c10d
    rendezvous, so init_process_group reads them from there. We must NOT set
    MASTER_ADDR/PORT by hand here — that would fight the rendezvous.
    """
    # "gloo" is the CPU backend. Use "nccl" if you move this to GPUs.
    dist.init_process_group(backend="gloo")


def cleanup():
    dist.destroy_process_group()


# define model
#class NeuralNetwork(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.flatten = nn.Flatten()
#        self.linear_relu_stack = nn.Sequential(
#            nn.Linear(28 * 28, 512),
#            nn.ReLU(),
#            nn.Linear(512, 512),
#            nn.ReLU(),
#            nn.Linear(512, 256),
#            nn.ReLU(),
#            nn.Linear(256, 10),
#        )
#
#    def forward(self, x):
#        x = self.flatten(x)
#        logits = self.linear_relu_stack(x)
#        return logits


# backup model for real example
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x); x = F.relu(x)
        x = self.conv2(x); x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x); x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    # NOTE: with log_softmax use nn.NLLLoss() instead of CrossEntropyLoss.


def train(dataloader, model, loss_fn, optimizer, device):
    """One epoch over this rank's shard. Returns local sums for reduction."""
    model.train()
    running_loss, correct, seen = 0.0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()           # DDP all-reduces gradients here
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        seen += y.size(0)

    return running_loss, correct, seen, len(dataloader)


def test(dataloader, model, loss_fn, device):
    """Full test set (rank 0 only). Returns (avg_test_loss, test_accuracy_%)."""
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return test_loss / len(dataloader), 100 * correct / len(dataloader.dataset)


def main():
    # Set thread counts BEFORE init_process_group (set_num_interop_threads
    # errors once the interop pool has been touched).
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = "cpu"

    if rank == 0:
        print(f"Using {device} device, world_size={world_size}", flush=True)

    # Each pod has its own container filesystem (no shared data volume), so
    # every rank downloads its own copy of MNIST into /app/data.
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

    batch_size = 64
    train_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)  # rank 0 evaluates the whole thing

    model = NeuralNetwork().to(device)
    ddp_model = DDP(model)
    #loss_fn = nn.CrossEntropyLoss()  # NOTE: no log_softmax in the model, so use CrossEntropyLoss
    loss_fn = nn.NLLLoss()  # NOTE: with log_softmax use nn.NLLLoss() instead of CrossEntropyLoss.
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)

    writer = None
    if rank == 0:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f"runs/ddp_{run_id}")  # -> /app/runs (mounted)
        images, _ = next(iter(train_dataloader))
        writer.add_graph(model, images.to(device))
        print(ddp_model, flush=True)

    epochs = 15
    total_train_time = 0.0

    for t in range(epochs):
        train_sampler.set_epoch(t)
        if rank == 0:
            print(f"Epoch {t+1}\n-------------------------------", flush=True)

        t0 = time.perf_counter()
        rloss, rcorrect, rseen, rbatches = train(train_dataloader, ddp_model, loss_fn, optimizer, device)
        total_train_time += time.perf_counter() - t0

        # Reduce TRAIN metrics across all ranks -> full-train-set numbers
        # (so train accuracy is comparable to the baseline, not just a shard).
        tm = torch.tensor([rloss, rcorrect, rseen, rbatches], dtype=torch.float64)
        dist.all_reduce(tm, op=dist.ReduceOp.SUM)
        g_loss, g_correct, g_seen, g_batches = tm.tolist()
        train_loss = g_loss / g_batches
        train_acc = 100 * g_correct / g_seen

        # TEST on rank 0 over the full set; others wait at the barrier.
        if rank == 0:
            test_loss, test_acc = test(test_dataloader, ddp_model, loss_fn, device)
        dist.barrier()

        if rank == 0:
            print(f"train: loss {train_loss:>7f} acc {train_acc:>0.1f}%  |  "
                  f"test: loss {test_loss:>7f} acc {test_acc:>0.1f}%", flush=True)
            writer.add_scalar("Loss/train", train_loss, t)
            writer.add_scalar("Loss/test", test_loss, t)
            writer.add_scalar("Accuracy/train", train_acc, t)
            writer.add_scalar("Accuracy/test", test_acc, t)
            writer.add_scalar("Accuracy/test_over_training_time", test_acc, int(round(total_train_time)))

    if rank == 0:
        print("Done!", flush=True)
        print(f"Total time spent training: {total_train_time:.2f} s", flush=True)
        writer.add_scalar("Time/total_training_seconds", total_train_time, 0)

        # save a file INSIDE it rather than to that path directly.
        os.makedirs("models", exist_ok=True)
        torch.save(ddp_model.module.state_dict(), "models/ddp_model.pth")
        print("Saved PyTorch Model State to ddp_model.pth", flush=True)

        writer.flush()
        writer.close()

        # ---- single-sample sanity check ----
        classes = [str(i) for i in range(10)]
        ddp_model.eval()
        x, y = test_data[0][0], test_data[0][1]
        x = x.unsqueeze(0).to(device)  # [1,28,28] -> [1,1,28,28]; required by the CNN backup
        with torch.no_grad():
            pred = ddp_model(x)
            print(f'Predicted: "{classes[pred[0].argmax(0)]}", Actual: "{classes[y]}"', flush=True)

    cleanup()


if __name__ == "__main__":
    # No mp.spawn: torchrun already started one process per pod and set up the
    # environment. We just run main() directly.
    main()