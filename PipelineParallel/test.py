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
# SHARED TENSORBOARD TAGS (identical in baseline / DDP / pipeline):
#   Loss/train, Loss/test, Accuracy/train, Accuracy/test   (x = epoch)
#   Accuracy/test_over_training_time                       (x = train seconds)
#   Time/total_training_seconds                            (single value)
# Local test version: 4 processes via mp.spawn, one pinned core each.
# Run with:  taskset -c 0,2,4,6 python3 pipeline_test.py
# All scalar logging happens on the LAST stage (that's where output/loss are).
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


# -------------------------------------------------------------------------
# Model split into 4 sequential stages (one per process). Together they form
# the same 784 -> 512 -> 512 -> 256 -> 10 MLP as the baseline / DDP versions.
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


def evaluate(stage_module, num_stages, test_dataloader, loss_fn, device, is_last):
    """Per-epoch held-out test.

    COLLECTIVE: every rank must call this (it all-gathers the stage weights).
    The last rank reassembles the full model and evaluates it; the others
    just contribute their weights and return (None, None).
    """
    cpu_state = {k: v.cpu() for k, v in stage_module.state_dict().items()}
    gathered = [None] * num_stages
    dist.all_gather_object(gathered, cpu_state)

    if not is_last:
        return None, None

    full = [build_stage_module(i) for i in range(num_stages)]
    for i, m in enumerate(full):
        m.load_state_dict(gathered[i])
        m.eval()

    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            x = X.to(device)
            for m in full:
                x = m(x)
            test_loss += loss_fn(x, y).item()
            correct += (x.argmax(1) == y).sum().item()
    return test_loss / len(test_dataloader), 100 * correct / len(test_dataloader.dataset)


def main(rank, world_size):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pin_to_core(rank, world_size)
    setup(rank, world_size)

    num_stages = world_size
    stage_index = rank
    is_last = stage_index == num_stages - 1
    device = torch.device("cpu")

    # ---- build only THIS process's stage + its optimizer ----
    stage_module = build_stage_module(stage_index).to(device)
    optimizer = torch.optim.SGD(stage_module.parameters(), lr=1e-3)
    stage = PipelineStage(stage_module, stage_index, num_stages, device)

    n_microbatches = 4  # more microbatches -> smaller pipeline bubble
    loss_fn = nn.CrossEntropyLoss()
    schedule = ScheduleGPipe(stage, n_microbatches=n_microbatches, loss_fn=loss_fn)

    training_data = datasets.MNIST(root="data", train=True, download=False, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=False, transform=ToTensor())

    batch_size = 64
    # drop_last keeps every batch divisible by n_microbatches (64 / 4 = 16).
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    writer = None
    if is_last:
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(f"runs/pipeline_{run_id}")
        # model graph: log the reassembled full architecture (fresh weights are
        # fine, add_graph only records structure)
        full_for_graph = nn.Sequential(*[build_stage_module(i) for i in range(num_stages)])
        images, _ = next(iter(train_dataloader))
        writer.add_graph(full_for_graph, images)
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
                losses = []
                output = schedule.step(target=y, losses=losses)
                running_loss += torch.stack(losses).mean().item()
                correct += (output.argmax(1) == y).sum().item()
                seen += y.size(0)
            else:
                schedule.step()

            optimizer.step()
            optimizer.zero_grad()

        total_train_time += time.perf_counter() - t0

        # per-epoch held-out test (collective; last rank computes the numbers)
        test_loss, test_acc = evaluate(stage_module, num_stages, test_dataloader, loss_fn, device, is_last)

        if is_last:
            train_loss = running_loss / len(train_dataloader)
            train_acc = 100 * correct / seen
            print(f"train: loss {train_loss:>7f} acc {train_acc:>0.1f}%  |  "
                  f"test: loss {test_loss:>7f} acc {test_acc:>0.1f}%", flush=True)
            writer.add_scalar("Loss/train", train_loss, t)
            writer.add_scalar("Loss/test", test_loss, t)
            writer.add_scalar("Accuracy/train", train_acc, t)
            writer.add_scalar("Accuracy/test", test_acc, t)
            writer.add_scalar("Accuracy/test_over_training_time", test_acc, int(round(total_train_time)))

    # ---- final gather for save + single-sample prediction (collective) ----
    cpu_state = {k: v.cpu() for k, v in stage_module.state_dict().items()}
    gathered = [None] * num_stages
    dist.all_gather_object(gathered, cpu_state)

    if is_last:
        print("Done!", flush=True)
        print(f"Total time spent training: {total_train_time:.2f} s", flush=True)
        writer.add_scalar("Time/total_training_seconds", total_train_time, 0)

        torch.save({f"stage{i}": gathered[i] for i in range(num_stages)}, "model_pipeline.pth")
        print("Saved pipeline model to model_pipeline.pth", flush=True)

        # single-sample sanity check on the reassembled model
        classes = [str(i) for i in range(10)]
        full = [build_stage_module(i) for i in range(num_stages)]
        for i, m in enumerate(full):
            m.load_state_dict(gathered[i])
            m.eval()
        x = test_data[0][0].unsqueeze(0).to(device)  # [1,28,28] -> [1,1,28,28]
        with torch.no_grad():
            for m in full:
                x = m(x)
            print(f'Predicted: "{classes[x[0].argmax(0)]}", '
                  f'Actual: "{classes[test_data[0][1]]}"', flush=True)

        writer.flush()
        writer.close()

    cleanup()


if __name__ == "__main__":
    # download once before spawning (shared filesystem -> avoid a 4-way race)
    datasets.MNIST(root="data", train=True, download=True)
    datasets.MNIST(root="data", train=False, download=True)

    world_size = 4
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)