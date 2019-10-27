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

import keras
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

print(f"keras Version: {keras.__version__}")

# The MNIST dataset comes pre-loaded in Keras, in the form of a set of four Numpy arrays:
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# `train_images` and `train_labels` form the "training set", the data that the model will learn from. The model will then be tested on the
# "test set", `test_images` and `test_labels`. Our images are encoded as Numpy arrays, and the labels are simply an array of digits, ranging
# from 0 to 9. There is a one-to-one correspondence between the images and the labels.
print(f"train_images.shape: {train_images.shape}")
print(f"test_images.shape: {test_images.shape}")

print(f"len(train_labels): {len(train_labels)}")
print(f"len(test_labels): {len(test_labels)}")

# Our workflow will be as follow: first we will present our neural network with the training data, `train_images` and `train_labels`. The
# network will then learn to associate images and labels. Finally, we will ask the network to produce predictions for `test_images`, and we
# will verify if these predictions match the labels from `test_labels`.
#
# Let's build our network -- again, remember that you aren't supposed to understand everything about this example just yet.
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

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
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Before training, we will preprocess our data by reshaping it into the shape that the network expects, and scaling it so that all values are in
# the `[0, 1]` interval. Previously, our training images for instance were stored in an array of shape `(60000, 28, 28)` of type `uint8` with
# values in the `[0, 255]` interval. We transform it into a `float32` array of shape `(60000, 28 * 28)` with values between 0 and 1.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# We also need to categorically encode the labels, a step which we explain in chapter 3:
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# We are now ready to train our network, which in Keras is done via a call to the `fit` method of the network:
# we "fit" the model to its training data.
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Two quantities are being displayed during training: the "loss" of the network over the training data, and the accuracy of the network over
# the training data.
# We quickly reach an accuracy of 0.989 (i.e. 98.9%) on the training data. Now let's check that our model performs well on the test set too:
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc:.6f}.")

# Our test set accuracy turns out to be 97.8% -- that's quite a bit lower than the training set accuracy.
# This gap between training accuracy and test accuracy is an example of "overfitting",
# the fact that machine learning models tend to perform worse on new data than on their training data.
# Overfitting will be a central topic in chapter 3.
#
# This concludes our very first example -- you just saw how we could build and a train a neural network to classify handwritten digits, in
# less than 20 lines of Python code. In the next chapter, we will go in detail over every moving piece we just previewed, and clarify what is really
# going on behind the scenes. You will learn about "tensors", the data-storing objects going into the network, about tensor operations, which
# layers are made of, and about gradient descent, which allows our network to learn from its training examples.
