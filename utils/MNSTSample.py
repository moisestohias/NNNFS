# imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets as MNIST
import torchvision.transforms as T

# Globals
epochs = 3
MBS = 64
BSTest = 1000
LR = 0.01
momentum = 0.5
log_interval = 10
torch.manual_seed(1)

# DataGetting
train_loader = DataLoader(MNIST('DS/', train=True, download=True, transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,)) ])), batch_size=MBS, shuffle=True)

test_loader = DataLoader(MNIST('DS/', train=False, download=True, transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,)) ])), batch_size=BSTest, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

# Model Def
class ConvNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class DenseNet(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.l1 = nn.Linear(784, 100)
        self.l2 = nn.Conv2d(100, 10)
    def forward(self, x):
        x = F.leaky_relu(self.l1(x.reshape(-1, 784)))
        return F.log_softmax(self.l2(x))

model = ConvNet()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(model.state_dict(), 'results/model.pth')
      torch.save(optimizer.state_dict(), 'results/optimizer.pth')
def test():
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = model(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

for epoch in range(1, epochs + 1):
  train(epoch)
  test()