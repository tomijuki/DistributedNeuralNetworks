import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter  # import tensorboard

import datetime
import time
import os

print(f"Number of CPU cores: {torch.get_num_threads()}")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(f"Number of CPU cores: {torch.get_num_threads()}")

# download training data from open datasets
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
# download test data from open datasets
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
# create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")


# define model
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


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# tensorboard setup
run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f"runs/non_distributed__{run_id}")

# log the model graph to tensorboard
dataiter = iter(train_dataloader)
images, labels = next(dataiter)
writer.add_graph(model, images.to(device))


def train(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # log loss every 100 batches to tensorboard
            global_step = epoch * len(dataloader) + batch
            writer.add_scalar("Training/Loss_per_batch", loss, global_step)


def test(dataloader, model, loss_fn, epoch, writer, train_time_s):
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

    # log test loss per epoch
    writer.add_scalar("Test/Average_loss_per_epoch", test_loss, epoch)
    # log test accuracy per epoch
    writer.add_scalar("Test/Accuracy_per_epoch", accuracy, epoch)

    # log accuracy against the CUMULATIVE time spent training.
    # The scalar step must be an int, so we use whole seconds of training
    # time as the x-axis (switch to int(train_time_s * 1000) for ms if your
    # epochs are sub-second and the points collide).
    step_seconds = int(round(train_time_s))
    writer.add_scalar("Test/Accuracy_over_training_time", accuracy, step_seconds)


epochs = 15
# print starting datetime
starting_time = datetime.datetime.now().strftime("%H:%M:%S")
print(f"Starting time: {starting_time}")

total_train_time = 0.0  # cumulative seconds spent inside train()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    # time only the training portion of the epoch
    t0 = time.perf_counter()
    train(train_dataloader, model, loss_fn, optimizer, t, writer)
    total_train_time += time.perf_counter() - t0

    # pass the cumulative training time so accuracy is logged against it
    test(test_dataloader, model, loss_fn, t, writer, total_train_time)
print("Done!")

# print elapsed (full run) time
ending_time = datetime.datetime.now().strftime("%H:%M:%S")
print(f"Ending time: {ending_time}")
print(
    "Elapsed time:",
    datetime.datetime.strptime(ending_time, "%H:%M:%S")
    - datetime.datetime.strptime(starting_time, "%H:%M:%S"),
)

# total time actually spent training (sum of the train() calls)
print(f"Total time spent training: {total_train_time:.2f} s")

# log the total training time for this run as a single number (one point at step 0)
writer.add_scalar("Time/Total_training_time_seconds", total_train_time, 0)

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

writer.flush()  # flush the tensorboard writer
writer.close()  # close the tensorboard writer


##load previously saved model and test it
##model = NeuralNetwork()
##model.load_state_dict(torch.load("model.pth", weights_only=True))
##model.to(device)
##test(test_dataloader, model, loss_fn)
##print("Done2!")


classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')