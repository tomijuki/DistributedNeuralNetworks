import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

import datetime
import time
import os

# =========================================================================
# LOCAL mp.spawn pipeline (4 processes, 1 per stage), for single-machine
# testing. Run pinned to 4 cores:
#
#     taskset -c 0,2,4,6 python pipeline_mp_spawn.py
#
# taskset limits the whole process tree to cores {0,2,4,6}; each rank then
# pins itself to ONE of them (rank i -> i-th allowed core) so the 4 stages
# don't contend for the same core. Using even cores avoids SMT siblings.
#
# SHARED TENSORBOARD TAGS (identical in baseline / DDP / pipeline):
#   Loss/train, Loss/test, Accuracy/train, Accuracy/test   (x = epoch)
#   Accuracy/test_over_training_time                       (x = train seconds)
#   Time/total_training_seconds                            (single value)
# Run dir runs/pipeline_* keeps it a separate line on the shared charts.
# Logging + model save happen on the LAST rank (the output stage).
# =========================================================================

WORLD_SIZE = 4
MASTER_ADDR = "localhost"
MASTER_PORT = "29500"


# -------------------------------------------------------------------------
# Distributed setup / teardown (manual, since there's no torchrun here)
# -------------------------------------------------------------------------
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    # gloo is the CPU backend; supports the point-to-point send/recv PP needs.
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def pin_to_core(rank):
    """Pin this rank to one core out of the taskset-allowed set.
    With `taskset -c 0,2,4,6` the allowed set is [0,2,4,6], so rank 0->0,
    1->2, 2->4, 3->6. Without taskset it falls back to cores 0..3."""
    allowed = sorted(os.sched_getaffinity(0))
    core = allowed[rank % len(allowed)]
    os.sched_setaffinity(0, {core})
    return core


# -------------------------------------------------------------------------
# Model, defined as 4 sequential STAGES (one per process).
# -------------------------------------------------------------------------
def build_stage_module(stage_index):
    if stage_index == 0:
        return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 512), nn.ReLU())
    elif stage_index == 1:
        return nn.Sequential(nn.Linear(512, 512), nn.ReLU())
    elif stage_index == 2:
        return nn.Sequential(nn.Linear(512, 256), nn.ReLU())
    elif stage_index == 3:
        return nn.Sequential(nn.Linear(256, 10))
    raise ValueError(f"no stage defined for index {stage_index}")


def gather_full_state(stage_module, num_stages):
    """Collective: EVERY rank must call this. Gathers each stage's weights so
    the last rank can reassemble the full model."""
    cpu_state = {k: v.cpu() for k, v in stage_module.state_dict().items()}
    gathered = [None] * num_stages
    dist.all_gather_object(gathered, cpu_state)
    return gathered


def evaluate(gathered, num_stages, test_dataloader, loss_fn, device):
    """Rebuild the full model from gathered stage weights and run the test set.
    Called on the last rank only."""
    full = [build_stage_module(i) for i in range(num_stages)]
    for i, m in enumerate(full):
        m.load_state_dict(gathered[i])
        m.eval()

    def full_forward(x):
        for m in full:
            x = m(x)
        return x

    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = full_forward(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(test_dataloader)
    test_acc = 100 * correct / len(test_dataloader.dataset)
    return test_loss, test_acc


# -------------------------------------------------------------------------
# Per-process entry point (mp.spawn passes `rank` as the first arg)
# -------------------------------------------------------------------------
def worker(rank, world_size):
    core = pin_to_core(rank)

    # One core per process -> one thread. Must be set before init_process_group.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    setup(rank, world_size)

    num_stages = world_size
    stage_index = rank
    is_last = stage_index == num_stages - 1
    device = torch.device("cpu")

    print(f"[rank {rank}] stage {stage_index} pinned to core {core}", flush=True)

    # ---- build only THIS process's slice of the model + its optimizer ----
    stage_module = build_stage_module(stage_index).to(device)
    optimizer = torch.optim.SGD(stage_module.parameters(), lr=1e-3)

    stage = PipelineStage(stage_module, stage_index, num_stages, device)

    n_microbatches = 4  # more microbatches -> smaller pipeline bubble
    loss_fn = nn.CrossEntropyLoss()
    schedule = ScheduleGPipe(stage, n_microbatches=n_microbatches, loss_fn=loss_fn)

    # ---- data: NOT sharded. Every rank iterates the same order. ----
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

    batch_size = 64
    # drop_last keeps every batch divisible by n_microbatches (64 / 4 = 16).
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    writer = None
    if is_last:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f"runs/pipeline_{run_id}")
        print(f"Pipeline: {num_stages} stages, {n_microbatches} microbatches, "
              f"batch {batch_size}", flush=True)

    epochs = 15
    total_train_time = 0.0

    for t in range(epochs):
        stage_module.train()
        if is_last:
            print(f"Epoch {t+1}\n-------------------------------", flush=True)

        running_loss, correct, seen = 0.0, 0, 0
        t0 = time.perf_counter()

        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)

            if stage_index == 0:
                schedule.step(X)
            elif is_last:
                losses = []  # one entry per microbatch
                output = schedule.step(target=y, losses=losses)
                running_loss += torch.stack(losses).mean().item()
                correct += (output.argmax(1) == y).type(torch.float).sum().item()
                seen += y.size(0)
            else:
                schedule.step()

            optimizer.step()
            optimizer.zero_grad()

        # Stop the TRAIN timer before evaluating (test is not timed).
        total_train_time += time.perf_counter() - t0

        # Per-epoch held-out test. gather_full_state is collective -> ALL ranks
        # call it every epoch; only the last rank evaluates and logs.
        gathered = gather_full_state(stage_module, num_stages)

        if is_last:
            test_loss, test_acc = evaluate(gathered, num_stages, test_dataloader, loss_fn, device)

            num_batches = len(train_dataloader)
            train_loss = running_loss / num_batches
            train_acc = 100 * correct / seen

            print(f"train: loss {train_loss:>7f} acc {train_acc:>0.1f}%  |  "
                  f"test: loss {test_loss:>7f} acc {test_acc:>0.1f}%", flush=True)

            writer.add_scalar("Loss/train", train_loss, t)
            writer.add_scalar("Loss/test", test_loss, t)
            writer.add_scalar("Accuracy/train", train_acc, t)
            writer.add_scalar("Accuracy/test", test_acc, t)
            writer.add_scalar("Accuracy/test_over_training_time", test_acc, int(round(total_train_time)))

    # ---- final gather (collective: all ranks) so the last rank can save ----
    gathered = gather_full_state(stage_module, num_stages)

    if is_last:
        print("Done!", flush=True)
        print(f"Total time spent training: {total_train_time:.2f} s", flush=True)
        writer.add_scalar("Time/total_training_seconds", total_train_time, 0)

        os.makedirs("models", exist_ok=True)
        torch.save({f"stage{i}": gathered[i] for i in range(num_stages)}, "models/pipeline_model.pth")
        print("Saved pipeline model to models/pipeline_model.pth", flush=True)

        writer.flush()
        writer.close()

    cleanup()


if __name__ == "__main__":
    # mp.spawn launches WORLD_SIZE processes, passing rank 0..WORLD_SIZE-1 as
    # the first arg to worker(); args=() supplies the rest.
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)