# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchsummary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='~/pytorch_datasets', train=True,
                                      download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=128,
                              shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='~/pytorch_datasets', train=False,
                                     download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=128,
                             shuffle=False, num_workers=2)

print(f"Pytorch Version: {torch.__version__}")


# ----------------------------- Currect Network --------------------------------------- #
# Our workflow will be as follow: first we will present our neural network with the training data, `train_images` and `train_labels`. The
# network will then learn to associate images and labels. Finally, we will ask the network to produce predictions for `test_images`, and we
# will verify if these predictions match the labels from `test_labels`.
#
# Let's build our network -- again, remember that you aren't supposed to understand everything about this example just yet.
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(28 * 28, 512)
    self.fc2 = nn.Linear(512, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


net = nn.DataParallel(Net())
net.to(device)
torchsummary.summary(net, (1, 28, 28))

# The core building block of neural networks is the "layer", a data-processing module which you can conceive as a "filter" for data. Some
# data comes in, and comes out in a more useful form. Precisely, layers extract _representations_ out of the data fed into them -- hopefully
# representations that are more meaningful for the problem at hand. Most of deep learning really consists of chaining together simple layers
# which will implement a form of progressive "data distillation". A deep learning model is like a sieve for data processing, made of a
# succession of increasingly refined data filters -- the "layers".
#
# Here our network consists of a sequence of two `Dense` layers, which are densely-connected (also called "fully-connected") neural layers.
# The second (and last) layer is a 10-way "softmax" layer, which means it will return an array of 10 probability scores (summing to 1). Each
# score will be the probability that the current digit image belongs to one of our 10 digit classes.
#
# To make our network ready for training, we need to pick three more things, as part of "compilation" step:
#
# * A loss function: the is how the network will be able to measure how good a job it is doing on its training data, and thus how it will be
# able to steer itself in the right direction.
# * An optimizer: this is the mechanism through which the network will update itself based on the data it sees and its loss function.
# * Metrics to monitor during training and testing. Here we will only care about accuracy (the fraction of the images that were correctly
# classified).
#
# The exact purpose of the loss function and the optimizer will be made clear throughout the next two chapters.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# We are now ready to train our network, which in Keras is done via a call to the `fit` method of the network:
# we "fit" the model to its training data.
for epoch in range(2):  # loop over the dataset multiple times

  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    if i % 50 == 0:
      print(f"Epoch: {epoch + 1:02d}  iter: {i * 128:05d} loss: {loss.item():.5f}.")

correct = 0
total = 0
with torch.no_grad():
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Accuracy of the currect network on the 10000 test images: {100 * correct / total}%.")
# ----------------------------- Currect Network --------------------------------------- #


# ----------------------------- Overfit Network --------------------------------------- #
class OverfitNet(nn.Module):
  def __init__(self):
    super(OverfitNet, self).__init__()
    self.fc1 = nn.Linear(28 * 28, 2048)
    self.fc2 = nn.Linear(2048, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


net = nn.DataParallel(OverfitNet())
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    if i % 50 == 0:
      print(f"Epoch: {epoch + 1:02d}  iter: {i * 128:05d} loss: {loss.item():.5f}.")

correct = 0
total = 0
with torch.no_grad():
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Accuracy of the overfit network on the 10000 test images: {100 * correct / total}%.")
# ----------------------------- Overfit Network --------------------------------------- #


# ----------------------------- Overfit Network --------------------------------------- #
class UnderfitNet(nn.Module):
  def __init__(self):
    super(UnderfitNet, self).__init__()
    self.fc1 = nn.Linear(28 * 28, 64)
    self.fc2 = nn.Linear(64, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


net = nn.DataParallel(UnderfitNet())
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    if i % 50 == 0:
      print(f"Epoch: {epoch + 1:02d}  iter: {i * 128:05d} loss: {loss.item():.5f}.")

correct = 0
total = 0
with torch.no_grad():
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Accuracy of the underfit network on the 10000 test images: {100 * correct / total}%.")
# ----------------------------- Underfit Network --------------------------------------- #
