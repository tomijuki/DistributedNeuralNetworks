import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

import datetime
import time


# -------------------------------------------------------------------------
# Distributed setup / teardown
# -------------------------------------------------------------------------
def setup():
    """Join the torchrun rendezvous (same as the DDP version).

    One process per pod; rank == pipeline stage index. gloo is the CPU
    backend and supports the point-to-point send/recv that PP relies on.
    """
    dist.init_process_group(backend="gloo")


def cleanup():
    dist.destroy_process_group()


# -------------------------------------------------------------------------
# Model, defined as 4 sequential STAGES (one per node).
# Each stage is a self-contained nn.Module. A node only ever builds its own.
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


# -------------------------------------------------------------------------
# Entry point (one process per pod, launched by torchrun)
# -------------------------------------------------------------------------
def main():
    setup()

    rank = dist.get_rank()
    num_stages = dist.get_world_size()  # 1 stage per rank
    stage_index = rank
    is_last = stage_index == num_stages - 1
    device = torch.device("cpu")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # ---- build only THIS node's slice of the model + its optimizer ----
    stage_module = build_stage_module(stage_index).to(device)
    optimizer = torch.optim.SGD(stage_module.parameters(), lr=1e-3)

    # Wrap the local module as a pipeline stage. Shapes for the inter-stage
    # send/recv are inferred automatically by the runtime.
    stage = PipelineStage(stage_module, stage_index, num_stages, device)

    # The schedule owns microbatch splitting, communication, and the backward
    # pass. Passing loss_fn here is what makes .step() run backward for us.
    n_microbatches = 4  # more microbatches -> smaller pipeline bubble
    loss_fn = nn.CrossEntropyLoss()
    schedule = ScheduleGPipe(stage, n_microbatches=n_microbatches, loss_fn=loss_fn)

    # ---- data: NOT sharded. Every rank iterates the same order so they agree
    # on batch boundaries; stage 0 feeds X, the last stage consumes y. ----
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

    batch_size = 64
    # drop_last keeps every batch divisible by n_microbatches (64 / 4 = 16).
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # tensorboard + prints live on the LAST stage (that's where output/loss are)
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

            # Drive the pipeline. Only stage 0 supplies input; only the last
            # stage supplies the target and receives the output.
            if stage_index == 0:
                schedule.step(X)
            elif is_last:
                losses = []  # one entry per microbatch
                output = schedule.step(target=y, losses=losses)
                running_loss += torch.stack(losses).mean().item()
                correct += (output.argmax(1) == y).sum().item()
                seen += y.size(0)
            else:
                schedule.step()

            # Each node updates only its own stage's parameters. The schedule
            # already populated .grad during its internal backward.
            optimizer.step()
            optimizer.zero_grad()

        total_train_time += time.perf_counter() - t0

        if is_last:
            num_batches = len(train_dataloader)
            train_loss = running_loss / num_batches
            train_acc = 100 * correct / seen
            print(f"train loss: {train_loss:>7f}  train acc: {train_acc:>0.1f}%", flush=True)
            writer.add_scalar("Train/Loss_per_epoch", train_loss, t)
            writer.add_scalar("Train/Accuracy_per_epoch", train_acc, t)
            writer.add_scalar("Train/Accuracy_over_training_time", train_acc, int(round(total_train_time)))

    # ---- held-out test ----
    # Gather every stage's weights onto the last rank, rebuild the full model
    # there, and evaluate normally. Exact, and avoids running backward at eval.
    cpu_state = {k: v.cpu() for k, v in stage_module.state_dict().items()}
    gathered = [None] * num_stages
    dist.all_gather_object(gathered, cpu_state)

    if is_last:
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
                pred = full_forward(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).sum().item()
        test_loss /= len(test_dataloader)
        test_acc = 100 * correct / len(test_data)
        print(f"\nTest Error: \n Accuracy: {test_acc:>0.1f}%, Avg loss: {test_loss:>8f}", flush=True)
        print(f"Total time spent training: {total_train_time:.2f} s", flush=True)

        writer.add_scalar("Test/Accuracy", test_acc, 0)
        writer.add_scalar("Time/Total_training_time_seconds", total_train_time, 0)

        # save the reassembled full model
        torch.save({f"stage{i}": gathered[i] for i in range(num_stages)}, "model_pipeline.pth")
        print("Saved pipeline model to model_pipeline.pth", flush=True)

        writer.flush()
        writer.close()

    cleanup()


if __name__ == "__main__":
    main()