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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,))])

torchvision.
trainset = torchvision.datasets.MNIST(root='~/pytorch_datasets', train=True,
                                      download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=128,
                              shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='~/pytorch_datasets', train=False,
                                     download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=128,
                             shuffle=False, num_workers=2)

print(f"Pytorch Version: {torch.__version__}")

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# Let's display the architecture of our convnet so far:

# In[3]:


model.summary()


# You can see above that the output of every `Conv2D` and `MaxPooling2D` layer is a 3D tensor of shape `(height, width, channels)`. The width 
# and height dimensions tend to shrink as we go deeper in the network. The number of channels is controlled by the first argument passed to 
# the `Conv2D` layers (e.g. 32 or 64).
# 
# The next step would be to feed our last output tensor (of shape `(3, 3, 64)`) into a densely-connected classifier network like those you are 
# already familiar with: a stack of `Dense` layers. These classifiers process vectors, which are 1D, whereas our current output is a 3D tensor. 
# So first, we will have to flatten our 3D outputs to 1D, and then add a few `Dense` layers on top:

# In[4]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# We are going to do 10-way classification, so we use a final layer with 10 outputs and a softmax activation. Now here's what our network 
# looks like:

# In[5]:


model.summary()


# As you can see, our `(3, 3, 64)` outputs were flattened into vectors of shape `(576,)`, before going through two `Dense` layers.
# 
# Now, let's train our convnet on the MNIST digits. We will reuse a lot of the code we have already covered in the MNIST example from Chapter 
# 2.

# In[6]:


from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[7]:


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)


# Let's evaluate the model on the test data:

# In[8]:


test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[9]:


test_acc


# While our densely-connected network from Chapter 2 had a test accuracy of 97.8%, our basic convnet has a test accuracy of 99.3%: we 
# decreased our error rate by 68% (relative). Not bad! 
