#set environment variable

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf

from network import Network
from fully_connected_layer import FcLayer
from activation import tanh, tanh_prime
from activation_layer import ActivationLayer
from  loss import mse, mse_prime

 #load Mnist from server
(mnist_train, mnist_test), mnist_data = tfds.load("mnist", split=['train', 'test'],
                                                   as_supervised=True, shuffle_files=True,
                                                     with_info=True,)
 
assert isinstance(mnist_train, tf.data.Dataset)

#normalizing img
#train
def normalize_img(image, label): 
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

mnist_train = mnist_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
mnist_train = mnist_train.cache()
mnist_train = mnist_train.shuffle(mnist_data.splits["train"].num_examples)
mnist_train = mnist_train.batch(128)
mnist_train = mnist_train.prefetch(tf.data.AUTOTUNE)
x_train, y_train = [], []
training_data = []
for images, labels in mnist_train:
    images = tf.reshape(images, (images.shape[0], -1)) # flatten the images
    labels = tf.one_hot(labels, depth=10) # one hot encoding the labels
    images, labels = images.numpy(), labels.numpy()
    x_train.extend(images)
    y_train.extend(labels)
#test
mnist_test = mnist_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
mnist_test = mnist_test.batch(128)
mnist_test = mnist_test.cache()
mnist_test = mnist_test.prefetch(tf.data.AUTOTUNE)
x_test, y_test = [], []

for image, label in mnist_test:
   image = tf.reshape(image, (image.shape[0], -1))
   label = tf.one_hot(label, depth=10)
   image, label = image.numpy(), label.numpy()
   x_test.extend(image)
   y_test.extend(label)

# Network, diese tolle graphik, die man überall sieht mit den layern und neuronen und so
net = Network()
net.add(FcLayer(784, 128))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100) layer 1
net.add(ActivationLayer(tanh, tanh_prime)) # or relu and relu_prime
net.add(FcLayer(128, 64))                   # input_shape=(1, 100)      ;   output_shape=(1, 50) layer 2

net.add(ActivationLayer(tanh, tanh_prime))   
net.add(FcLayer(64, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10) layer 3
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=50, learning_rate=0.01)

# evaluate the network on the test set
predictions = net.predict(x_test)
for i in range(5):
  print(f"Predicted: {np.argmax(predictions[i])}, True: {np.argmax(labels[i])}")
# evaluate against y_test
print("\n")
#print("predicted values : ")
#print(out, end="\n")
#print("true values : ")
#print(labels[:3], end="\n")
