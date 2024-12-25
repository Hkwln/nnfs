#set environment variable

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from network import Network
from fully_connected_layer import FcLayer
from activation import tanh, tanh_prime, relu, relu_prime
from activation_layer import ActivationLayer
from  loss import mse, mse_prime

#load Mnist data from Tensorflow_datasets
(mnist_train, mnist_test), mnist_data = tfds.load("mnist", split=['train', 'test'],
                                                   as_supervised=True, shuffle_files=True,
                                                     with_info=True,)

def normalize_img(image, label):
  image = np.array(image) / 255.0
  label = np.eye(10)[label]  # One-hot encoding
  return image, label

#prepare training data
def prepare_data(dataset):
  images, labels = [], []
  for image, label in dataset:
    image, label = normalize_img(image, label)
    image = np.array(image)  # Ensure image is a numpy array
    image = image.reshape(-1)  # Reshape to (28, 28, 1)
    images.append(image)
    labels.append(label)
  return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

# prepare train data
mnist_train = mnist_train.cache()
mnist_train = mnist_train.shuffle(mnist_data.splits["train"].num_examples)
mnist_train = mnist_train.batch(128)
mnist_train = mnist_train.unbatch()
x_train, y_train = prepare_data(mnist_train)

#prepare test data
mnist_test = mnist_test.batch(128)
mnist_test = mnist_test.cache()
mnist_test = mnist_test.unbatch()
x_test, y_test = prepare_data(mnist_test)

# Network, diese tolle graphik, die man überall sieht mit den layern und neuronen und so
net = Network()

net.add(FcLayer(784, 128))                 # layer 1
net.add(ActivationLayer(tanh, tanh_prime)) # tanh or tanh_prime relu and relu_prime 
net.add(FcLayer(128, 64))                  # layer 2
net.add(ActivationLayer(tanh, tanh_prime))   
net.add(FcLayer(64, 10))                    # layer 3
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
#load weights if they exist
if os.path.exists('weights.npy'):
  net.load_weights('weights.npy')
net.fit(x_train, y_train, epochs=30, learning_rate=0.01)

# evaluate the network on the test set
predictions = net.predict(x_test)
#save weights
net.save_weights('weights.npy')

for i in range(10):
    print(f"Predicted: {np.argmax(predictions[i])}, True: {np.argmax(y_test[i])}")  # Explicitly convert to NumPy array
