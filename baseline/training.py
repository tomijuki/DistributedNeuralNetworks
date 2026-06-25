import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import datetime
import time
import os

# =========================================================================
# SHARED TENSORBOARD TAGS (identical in baseline / DDP / pipeline so the
# curves overlay on the same charts):
#   Loss/train                        (x = epoch)
#   Loss/test                         (x = epoch)
#   Accuracy/train                    (x = epoch)
#   Accuracy/test                     (x = epoch)
#   Accuracy/test_over_training_time  (x = cumulative training seconds)
#   Time/total_training_seconds       (single value at step 0)
# Run dirs stay DISTINCT (runs/baseline_*, runs/ddp_*, runs/pipeline_*);
# that's what makes TensorBoard draw the three as separate lines per chart.
# =========================================================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device = "cpu"


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
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    # NOTE: with log_softmax use nn.NLLLoss() instead of CrossEntropyLoss.


def train(dataloader, model, loss_fn, optimizer, device):
    """Run one epoch. Returns (avg_train_loss, train_accuracy_%)."""
    model.train()
    running_loss, correct, seen = 0.0, 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        seen += y.size(0)

    return running_loss / len(dataloader), 100 * correct / seen


def test(dataloader, model, loss_fn, device):
    """Evaluate on the full test set. Returns (avg_test_loss, test_accuracy_%)."""
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
    print(f"Number of CPU threads: {torch.get_num_threads()}")
    print(f"Using {device} device")

    training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"runs/baseline_{run_id}")
    images, _ = next(iter(train_dataloader))
    writer.add_graph(model, images.to(device))

    epochs = 15
    total_train_time = 0.0

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        t0 = time.perf_counter()
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
        total_train_time += time.perf_counter() - t0

        test_loss, test_acc = test(test_dataloader, model, loss_fn, device)

        print(f"train: loss {train_loss:>7f} acc {train_acc:>0.1f}%  |  "
              f"test: loss {test_loss:>7f} acc {test_acc:>0.1f}%")

        writer.add_scalar("Loss/train", train_loss, t)
        writer.add_scalar("Loss/test", test_loss, t)
        writer.add_scalar("Accuracy/train", train_acc, t)
        writer.add_scalar("Accuracy/test", test_acc, t)
        writer.add_scalar("Accuracy/test_over_training_time", test_acc, int(round(total_train_time)))

    print("Done!")
    print(f"Total time spent training: {total_train_time:.2f} s")
    writer.add_scalar("Time/total_training_seconds", total_train_time, 0)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_model.pth")
    print("Saved PyTorch Model State to baseline_model.pth")
    writer.flush()
    writer.close()

    # ---- single-sample sanity check ----
    classes = [str(i) for i in range(10)]
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    x = x.unsqueeze(0).to(device)  # [1,28,28] -> [1,1,28,28]; required by the CNN backup
    with torch.no_grad():
        pred = model(x)
        print(f'Predicted: "{classes[pred[0].argmax(0)]}", Actual: "{classes[y]}"')


if __name__ == "__main__":
    main()