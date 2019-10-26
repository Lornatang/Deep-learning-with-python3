#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
keras.__version__


# # Using a pre-trained convnet
# 
# This notebook contains the code sample found in Chapter 5, Section 3 of [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff). Note that the original text features far more content, in particular further explanations and figures: in this notebook, you will only find source code and related comments.
# 
# ----
# 
# A common and highly effective approach to deep learning on small image datasets is to leverage a pre-trained network. A pre-trained network 
# is simply a saved network previously trained on a large dataset, typically on a large-scale image classification task. If this original 
# dataset is large enough and general enough, then the spatial feature hierarchy learned by the pre-trained network can effectively act as a 
# generic model of our visual world, and hence its features can prove useful for many different computer vision problems, even though these 
# new problems might involve completely different classes from those of the original task. For instance, one might train a network on 
# ImageNet (where classes are mostly animals and everyday objects) and then re-purpose this trained network for something as remote as 
# identifying furniture items in images. Such portability of learned features across different problems is a key advantage of deep learning 
# compared to many older shallow learning approaches, and it makes deep learning very effective for small-data problems.
# 
# In our case, we will consider a large convnet trained on the ImageNet dataset (1.4 million labeled images and 1000 different classes). 
# ImageNet contains many animal classes, including different species of cats and dogs, and we can thus expect to perform very well on our cat 
# vs. dog classification problem.
# 
# We will use the VGG16 architecture, developed by Karen Simonyan and Andrew Zisserman in 2014, a simple and widely used convnet architecture 
# for ImageNet. Although it is a bit of an older model, far from the current state of the art and somewhat heavier than many other recent 
# models, we chose it because its architecture is similar to what you are already familiar with, and easy to understand without introducing 
# any new concepts. This may be your first encounter with one of these cutesie model names -- VGG, ResNet, Inception, Inception-ResNet, 
# Xception... you will get used to them, as they will come up frequently if you keep doing deep learning for computer vision.
# 
# There are two ways to leverage a pre-trained network: *feature extraction* and *fine-tuning*. We will cover both of them. Let's start with 
# feature extraction.

# ## Feature extraction
# 
# Feature extraction consists of using the representations learned by a previous network to extract interesting features from new samples. 
# These features are then run through a new classifier, which is trained from scratch.
# 
# As we saw previously, convnets used for image classification comprise two parts: they start with a series of pooling and convolution 
# layers, and they end with a densely-connected classifier. The first part is called the "convolutional base" of the model. In the case of 
# convnets, "feature extraction" will simply consist of taking the convolutional base of a previously-trained network, running the new data 
# through it, and training a new classifier on top of the output.
# 
# ![swapping FC classifiers](https://s3.amazonaws.com/book.keras.io/img/ch5/swapping_fc_classifier.png)
# 
# Why only reuse the convolutional base? Could we reuse the densely-connected classifier as well? In general, it should be avoided. The 
# reason is simply that the representations learned by the convolutional base are likely to be more generic and therefore more reusable: the 
# feature maps of a convnet are presence maps of generic concepts over a picture, which is likely to be useful regardless of the computer 
# vision problem at hand. On the other end, the representations learned by the classifier will necessarily be very specific to the set of 
# classes that the model was trained on -- they will only contain information about the presence probability of this or that class in the 
# entire picture. Additionally, representations found in densely-connected layers no longer contain any information about _where_ objects are 
# located in the input image: these layers get rid of the notion of space, whereas the object location is still described by convolutional 
# feature maps. For problems where object location matters, densely-connected features would be largely useless.
# 
# Note that the level of generality (and therefore reusability) of the representations extracted by specific convolution layers depends on 
# the depth of the layer in the model. Layers that come earlier in the model extract local, highly generic feature maps (such as visual 
# edges, colors, and textures), while layers higher-up extract more abstract concepts (such as "cat ear" or "dog eye"). So if your new 
# dataset differs a lot from the dataset that the original model was trained on, you may be better off using only the first few layers of the 
# model to do feature extraction, rather than using the entire convolutional base.
# 
# In our case, since the ImageNet class set did contain multiple dog and cat classes, it is likely that it would be beneficial to reuse the 
# information contained in the densely-connected layers of the original model. However, we will chose not to, in order to cover the more 
# general case where the class set of the new problem does not overlap with the class set of the original model.

# Let's put this in practice by using the convolutional base of the VGG16 network, trained on ImageNet, to extract interesting features from 
# our cat and dog images, and then training a cat vs. dog classifier on top of these features.
# 
# The VGG16 model, among others, comes pre-packaged with Keras. You can import it from the `keras.applications` module. Here's the list of 
# image classification models (all pre-trained on the ImageNet dataset) that are available as part of `keras.applications`:
# 
# * Xception
# * InceptionV3
# * ResNet50
# * VGG16
# * VGG19
# * MobileNet
# 
# Let's instantiate the VGG16 model:

# In[2]:


from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


# We passed three arguments to the constructor:
# 
# * `weights`, to specify which weight checkpoint to initialize the model from
# * `include_top`, which refers to including or not the densely-connected classifier on top of the network. By default, this 
# densely-connected classifier would correspond to the 1000 classes from ImageNet. Since we intend to use our own densely-connected 
# classifier (with only two classes, cat and dog), we don't need to include it.
# * `input_shape`, the shape of the image tensors that we will feed to the network. This argument is purely optional: if we don't pass it, 
# then the network will be able to process inputs of any size.
# 
# Here's the detail of the architecture of the VGG16 convolutional base: it's very similar to the simple convnets that you are already 
# familiar with.

# In[3]:


conv_base.summary()


# The final feature map has shape `(4, 4, 512)`. That's the feature on top of which we will stick a densely-connected classifier.
# 
# At this point, there are two ways we could proceed: 
# 
# * Running the convolutional base over our dataset, recording its output to a Numpy array on disk, then using this data as input to a 
# standalone densely-connected classifier similar to those you have seen in the first chapters of this book. This solution is very fast and 
# cheap to run, because it only requires running the convolutional base once for every input image, and the convolutional base is by far the 
# most expensive part of the pipeline. However, for the exact same reason, this technique would not allow us to leverage data augmentation at 
# all.
# * Extending the model we have (`conv_base`) by adding `Dense` layers on top, and running the whole thing end-to-end on the input data. This 
# allows us to use data augmentation, because every input image is going through the convolutional base every time it is seen by the model. 
# However, for this same reason, this technique is far more expensive than the first one.
# 
# We will cover both techniques. Let's walk through the code required to set-up the first one: recording the output of `conv_base` on our 
# data and using these outputs as inputs to a new model.
# 
# We will start by simply running instances of the previously-introduced `ImageDataGenerator` to extract images as Numpy arrays as well as 
# their labels. We will extract features from these images simply by calling the `predict` method of the `conv_base` model.

# In[4]:


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)


# The extracted features are currently of shape `(samples, 4, 4, 512)`. We will feed them to a densely-connected classifier, so first we must 
# flatten them to `(samples, 8192)`:

# In[5]:


train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


# At this point, we can define our densely-connected classifier (note the use of dropout for regularization), and train it on the data and 
# labels that we just recorded:

# In[6]:


from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))


# Training is very fast, since we only have to deal with two `Dense` layers -- an epoch takes less than one second even on CPU.
# 
# Let's take a look at the loss and accuracy curves during training:

# In[7]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# 
# We reach a validation accuracy of about 90%, much better than what we could achieve in the previous section with our small model trained from 
# scratch. However, our plots also indicate that we are overfitting almost from the start -- despite using dropout with a fairly large rate. 
# This is because this technique does not leverage data augmentation, which is essential to preventing overfitting with small image datasets.
# 
# Now, let's review the second technique we mentioned for doing feature extraction, which is much slower and more expensive, but which allows 
# us to leverage data augmentation during training: extending the `conv_base` model and running it end-to-end on the inputs. Note that this 
# technique is in fact so expensive that you should only attempt it if you have access to a GPU: it is absolutely intractable on CPU. If you 
# cannot run your code on GPU, then the previous technique is the way to go.
# 
# Because models behave just like layers, you can add a model (like our `conv_base`) to a `Sequential` model just like you would add a layer. 
# So you can do the following:

# In[8]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# This is what our model looks like now:

# In[9]:


model.summary()


# As you can see, the convolutional base of VGG16 has 14,714,688 parameters, which is very large. The classifier we are adding on top has 2 
# million parameters.
# 
# Before we compile and train our model, a very important thing to do is to freeze the convolutional base. "Freezing" a layer or set of 
# layers means preventing their weights from getting updated during training. If we don't do this, then the representations that were 
# previously learned by the convolutional base would get modified during training. Since the `Dense` layers on top are randomly initialized, 
# very large weight updates would be propagated through the network, effectively destroying the representations previously learned.
# 
# In Keras, freezing a network is done by setting its `trainable` attribute to `False`:

# In[10]:


print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))


# In[11]:


conv_base.trainable = False


# In[12]:


print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))


# With this setup, only the weights from the two `Dense` layers that we added will be trained. That's a total of four weight tensors: two per 
# layer (the main weight matrix and the bias vector). Note that in order for these changes to take effect, we must first compile the model. 
# If you ever modify weight trainability after compilation, you should then re-compile the model, or these changes would be ignored.
# 
# Now we can start training our model, with the same data augmentation configuration that we used in our previous example:

# In[13]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)


# In[14]:


model.save('cats_and_dogs_small_3.h5')


# Let's plot our results again:

# In[15]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# As you can see, we reach a validation accuracy of about 96%. This is much better than our small convnet trained from scratch.

# ## Fine-tuning
# 
# Another widely used technique for model reuse, complementary to feature extraction, is _fine-tuning_. 
# Fine-tuning consists in unfreezing a few of the top layers 
# of a frozen model base used for feature extraction, and jointly training both the newly added part of the model (in our case, the 
# fully-connected classifier) and these top layers. This is called "fine-tuning" because it slightly adjusts the more abstract 
# representations of the model being reused, in order to make them more relevant for the problem at hand.
# 
# ![fine-tuning VGG16](https://s3.amazonaws.com/book.keras.io/img/ch5/vgg16_fine_tuning.png)

# We have stated before that it was necessary to freeze the convolution base of VGG16 in order to be able to train a randomly initialized 
# classifier on top. For the same reason, it is only possible to fine-tune the top layers of the convolutional base once the classifier on 
# top has already been trained. If the classified wasn't already trained, then the error signal propagating through the network during 
# training would be too large, and the representations previously learned by the layers being fine-tuned would be destroyed. Thus the steps 
# for fine-tuning a network are as follow:
# 
# * 1) Add your custom network on top of an already trained base network.
# * 2) Freeze the base network.
# * 3) Train the part you added.
# * 4) Unfreeze some layers in the base network.
# * 5) Jointly train both these layers and the part you added.
# 
# We have already completed the first 3 steps when doing feature extraction. Let's proceed with the 4th step: we will unfreeze our `conv_base`, 
# and then freeze individual layers inside of it.
# 
# As a reminder, this is what our convolutional base looks like:

# In[17]:


conv_base.summary()


# 
# We will fine-tune the last 3 convolutional layers, which means that all layers up until `block4_pool` should be frozen, and the layers 
# `block5_conv1`, `block5_conv2` and `block5_conv3` should be trainable.
# 
# Why not fine-tune more layers? Why not fine-tune the entire convolutional base? We could. However, we need to consider that:
# 
# * Earlier layers in the convolutional base encode more generic, reusable features, while layers higher up encode more specialized features. It is 
# more useful to fine-tune the more specialized features, as these are the ones that need to be repurposed on our new problem. There would 
# be fast-decreasing returns in fine-tuning lower layers.
# * The more parameters we are training, the more we are at risk of overfitting. The convolutional base has 15M parameters, so it would be 
# risky to attempt to train it on our small dataset.
# 
# Thus, in our situation, it is a good strategy to only fine-tune the top 2 to 3 layers in the convolutional base.
# 
# Let's set this up, starting from where we left off in the previous example:

# In[18]:


conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# Now we can start fine-tuning our network. We will do this with the RMSprop optimizer, using a very low learning rate. The reason for using 
# a low learning rate is that we want to limit the magnitude of the modifications we make to the representations of the 3 layers that we are 
# fine-tuning. Updates that are too large may harm these representations.
# 
# Now let's proceed with fine-tuning:

# In[19]:


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)


# In[20]:


model.save('cats_and_dogs_small_4.h5')


# Let's plot our results using the same plotting code as before:

# In[22]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# 
# These curves look very noisy. To make them more readable, we can smooth them by replacing every loss and accuracy with exponential moving 
# averages of these quantities. Here's a trivial utility function to do this:

# In[24]:


def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# 
# These curves look much cleaner and more stable. We are seeing a nice 1% absolute improvement.
# 
# Note that the loss curve does not show any real improvement (in fact, it is deteriorating). You may wonder, how could accuracy improve if the 
# loss isn't decreasing? The answer is simple: what we display is an average of pointwise loss values, but what actually matters for accuracy 
# is the distribution of the loss values, not their average, since accuracy is the result of a binary thresholding of the class probability 
# predicted by the model. The model may still be improving even if this isn't reflected in the average loss.
# 
# We can now finally evaluate this model on the test data:

# In[27]:


test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)


# 
# Here we get a test accuracy of 97%. In the original Kaggle competition around this dataset, this would have been one of the top results. 
# However, using modern deep learning techniques, we managed to reach this result using only a very small fraction of the training data 
# available (about 10%). There is a huge difference between being able to train on 20,000 samples compared to 2,000 samples!

# ## Take-aways: using convnets with small datasets
# 
# Here's what you should take away from the exercises of these past two sections:
# 
# * Convnets are the best type of machine learning models for computer vision tasks. It is possible to train one from scratch even on a very 
# small dataset, with decent results.
# * On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when working with image 
# data.
# * It is easy to reuse an existing convnet on a new dataset, via feature extraction. This is a very valuable technique for working with 
# small image datasets.
# * As a complement to feature extraction, one may use fine-tuning, which adapts to a new problem some of the representations previously 
# learned by an existing model. This pushes performance a bit further.
# 
# Now you have a solid set of tools for dealing with image classification problems, in particular with small datasets.
