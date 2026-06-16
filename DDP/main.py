import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter #import tensorboard
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import datetime

# Initialize Distributed Process Group
# Use gloo 
dist.init_process_group(backend='gloo')

# get rank and world size
RANK = int(os.environ['RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])


print(f"Rank: {RANK}, World Size: {WORLD_SIZE}")

torch.set_num_threads(1)
device = "cpu"

# Download data
if RANK == 0:
    datasets.MNIST(root="data", train=True, download=True)
    datasets.MNIST(root="data", train=False, download=True)
dist.barrier()  # Ensure that the data is downloaded before other processes proceed

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64

#Distributed Sampler: Splits the dataset into 4 uniqe chunks
train_sampler = DistributedSampler(training_data, num_replicas=WORLD_SIZE, rank=RANK)
train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)

# FOr testing, only the process with rank 0 will perform testing
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

# Wrap the model with DDP
model = DDP(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#Only rank 0 sets up tensorboard
writer = None
if RANK == 0:
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"runs/ddp_distributed_{run_id}")

def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    dataloader.sampler.set_epoch(epoch)  # Set epoch for shuffling with DistributedSampler

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0 and RANK == 0:
            print(f"[Rank 0] loss: {loss.item():>7f}  [{batch * len(X):>5d}]")
            global_step = (epoch * len(dataloader) + batch) * WORLD_SIZE
            writer.add_scalar("Training/Loss_per_batch", loss.item(), global_step)

def test(dataloader, model, loss_fn, epoch):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader.dataset)
    accuracy = 100 * correct / len(dataloader.dataset)
    print(f"[Rank 0] Test Error: \n Accuracy: {accuracy:.1f}%, Avg loss: {test_loss:.8f} \n")
    writer.add_scalar("Test/Average_loss_per_epoch", test_loss, epoch)
    writer.add_scalar("Test/Accuracy_per_epoch", accuracy, epoch)


epochs = 15
for t in range(epochs):
    if RANK == 0:
        print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t)

    # Only rank 0 performs testing
    if RANK == 0:
        test(test_dataloader, model, loss_fn, t)
    
# only rank 0 saves the model and flushes the tensorboard writer
if RANK == 0:
    torch.save(model.state_dict(), "DDPmodel.pth")
    print("Saved PyTorch Model State to DDPmodel.pth")
    writer.flush()
    writer.close()

dist.destroy_process_group()