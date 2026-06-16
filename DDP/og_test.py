import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter  # import tensorboard

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import datetime
import time
import os


# -------------------------------------------------------------------------
# Distributed setup / teardown
# -------------------------------------------------------------------------
def setup(rank, world_size):
    """Initialize the process group (the 'rendezvous')."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # "gloo" is the CPU backend. Use "nccl" if you move this to GPUs.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# -------------------------------------------------------------------------
# Model (unchanged)
# -------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# -------------------------------------------------------------------------
# Train / test loops
# (rank is passed in so only rank 0 prints / logs)
# -------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer, epoch, writer, rank, device, world_size):
    # With a DistributedSampler each rank only sees a shard of the data,
    # so report progress against this rank's shard size.
    size = len(dataloader.sampler)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation (DDP all-reduces the gradients here automatically)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0 and rank == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # log loss every 100 batches to tensorboard.
            # x world_size so the step approximates the number of samples seen
            # across all ranks (comparable to the non-distributed run).
            global_step = (epoch * len(dataloader) + batch) * world_size
            writer.add_scalar("Training/Loss_per_batch", loss, global_step)


def test(dataloader, model, loss_fn, epoch, writer, device, train_time_s):
    # Called on rank 0 only, over the full (non-sharded) test set.
    # All ranks hold identical weights after DDP's gradient sync, so
    # evaluating on rank 0 gives the correct accuracy.
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # log test loss per epoch to tensorboard
    writer.add_scalar("Test/Average_loss_per_epoch", test_loss, epoch)
    # log test accuracy per epoch
    writer.add_scalar("Test/Accuracy_per_epoch", accuracy, epoch)

    # log accuracy against the CUMULATIVE time spent training (rank 0's clock).
    # The scalar step must be an int, so we use whole seconds of training time
    # as the x-axis (switch to int(train_time_s * 1000) for ms if your epochs
    # are sub-second and the points collide).
    step_seconds = int(round(train_time_s))
    writer.add_scalar("Test/Accuracy_over_training_time", accuracy, step_seconds)



# -------------------------------------------------------------------------
# Per-process entry point (one call per spawned process)
# -------------------------------------------------------------------------
def main(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        print(f"Number of CPU threads: {torch.get_num_threads()}")
    # NOTE: 4 processes x 1 thread = 4 threads total. Bump this up if you have
    # spare cores per process (e.g. torch.set_num_threads(cores // world_size)).
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if rank == 0:
        print(f"Number of CPU threads: {torch.get_num_threads()}")

    device = "cpu"
    if rank == 0:
        print(f"Using {device} device")

    # Data was already downloaded in __main__, so download=False here avoids
    # 4 processes racing to write the same files.
    training_data = datasets.MNIST(
        root="data", train=True, download=False, transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=False, transform=ToTensor()
    )

    batch_size = 64

    # The DistributedSampler hands each of the 4 processes a disjoint shard of
    # the training set. (shuffle=False to match the original DataLoader; flip
    # it to True for better training, then call set_epoch each epoch.)
    # Accuracy: 71.5%, Avg loss: 1.866718 
    train_sampler = DistributedSampler(
        training_data, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, sampler=train_sampler
    )
    # Test set is NOT sharded: rank 0 evaluates over the whole thing.
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    if rank == 0:
        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

    # Build the model and wrap it in DDP. No device_ids for the CPU backend.
    model = NeuralNetwork().to(device)
    ddp_model = DDP(model)
    if rank == 0:
        print(ddp_model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

    # tensorboard setup (rank 0 only)
    writer = None
    if rank == 0:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f"runs/ddp_distributed_{run_id}")

        # log the model graph to tensorboard (use the raw module)
        dataiter = iter(train_dataloader)
        images, labels = next(dataiter)
        writer.add_graph(model, images.to(device))

    epochs = 15
    if rank == 0:
        starting_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"Starting time: {starting_time}")

    total_train_time = 0.0  # cumulative seconds spent inside train()

    for t in range(epochs):
        # Keeps shard ordering consistent per epoch (matters once shuffle=True).
        train_sampler.set_epoch(t)

        if rank == 0:
            print(f"Epoch {t+1}\n-------------------------------")

        # time only the training portion of the epoch
        t0 = time.perf_counter()
        train(train_dataloader, ddp_model, loss_fn, optimizer, t, writer, rank, device, world_size)
        total_train_time += time.perf_counter() - t0

        if rank == 0:
            # pass the cumulative training time so accuracy is logged against it
            test(test_dataloader, ddp_model, loss_fn, t, writer, device, total_train_time)

        # Make every rank wait for rank 0 to finish testing before the next
        # epoch's gradient all-reduce, so nobody races ahead.
        dist.barrier()

    if rank == 0:
        print("Done!")

        ending_time = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"Ending time: {ending_time}")
        print(
            "Elapsed time:",
            datetime.datetime.strptime(ending_time, "%H:%M:%S")
            - datetime.datetime.strptime(starting_time, "%H:%M:%S"),
        )

        # total time actually spent training (sum of the train() calls)
        print(f"Total time spent training: {total_train_time:.2f} s")

        # log the total training time for this run as a single number
        writer.add_scalar("Time/Total_training_time_seconds", total_train_time, 0)

        # Save the underlying module's state dict (strips the "module." prefix).
        torch.save(ddp_model.module.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

        writer.flush()  # flush the tensorboard writer
        writer.close()  # close the tensorboard writer

        # ---- single-sample sanity check ----
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        ddp_model.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            x = x.to(device)
            pred = ddp_model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')

    cleanup()


if __name__ == "__main__":
    # Download the dataset ONCE in the parent process before spawning, so the
    # 4 workers don't race to download/extract the same files.
    datasets.MNIST(root="data", train=True, download=True)
    datasets.MNIST(root="data", train=False, download=True)

    world_size = 4
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)