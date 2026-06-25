import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import datetime
import time
import os

# =========================================================================
# SHARED TENSORBOARD TAGS (identical in baseline / DDP / pipeline):
#   Loss/train, Loss/test, Accuracy/train, Accuracy/test   (x = epoch)
#   Accuracy/test_over_training_time                       (x = train seconds)
#   Time/total_training_seconds                            (single value)
# Local test version: 4 processes via mp.spawn, one pinned core each.
# Run with:  taskset -c 0,2,4,6 python3 ddp_test.py
# =========================================================================


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def pin_to_core(rank, world_size):
    """Pin rank i to the i-th allowed core (respects an outer taskset mask)."""
    try:
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) >= world_size:
            os.sched_setaffinity(0, {allowed[rank]})
    except (AttributeError, OSError):
        pass


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


def main(rank, world_size):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pin_to_core(rank, world_size)
    setup(rank, world_size)
    device = "cpu"

    if rank == 0:
        print(f"Using {device} device, world_size={world_size}", flush=True)

    training_data = datasets.MNIST(root="data", train=True, download=False, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=False, transform=ToTensor())

    batch_size = 64
    train_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)  # rank 0 evaluates the whole thing

    model = NeuralNetwork().to(device)
    ddp_model = DDP(model)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)

    writer = None
    if rank == 0:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f"runs/ddp_{run_id}")
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

        torch.save(ddp_model.module.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth", flush=True)
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
    # download once before spawning (shared filesystem -> avoid a 4-way race)
    datasets.MNIST(root="data", train=True, download=True)
    datasets.MNIST(root="data", train=False, download=True)

    world_size = 4
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)