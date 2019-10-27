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
import torchsummary
import torchvision
import torchvision.transforms as transforms

from utils.eval import accuracy
from utils.misc import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
  [transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='~/pytorch_datasets', train=True,
                                      download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=128,
                              shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='~/pytorch_datasets', train=False,
                                     download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=128,
                             shuffle=False, num_workers=2)

print(f"PyTorch Version: {torch.__version__}")


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.fc1 = nn.Linear(5 * 5 * 64, 512)
    self.fc2 = nn.Linear(512, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, 5 * 5 * 64)
    x = self.fc1(x)
    x = self.fc2(x)
    return x


net = nn.DataParallel(Net())
net.to(device)
torchsummary.summary(net, (1, 28, 28))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# We are now ready to train our network, which in Keras is done via a call to the `fit` method of the network:
# we "fit" the model to its training data.
for epoch in range(5):  # loop over the dataset multiple times
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  net.train()
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
    losses.update(loss.item(), inputs.size(0))
    top1.update(prec1, inputs.size(0))
    top5.update(prec5, inputs.size(0))

    loss.backward()
    optimizer.step()

    # print statistics
    if i % 5 == 0:
      print(f"Epoch [{epoch + 1}] [{i}/{len(trainloader)}]\t"
            f"Loss {loss.item():.4f}\t"
            f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
            f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})")

net.eval()
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
