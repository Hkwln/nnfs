import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
from network import Network
from fully_connected_layer import FcLayer
from activation import tanh, tanh_prime
from activation_layer import ActivationLayer
from  loss import mse, mse_prime

#from keras.dataset import mnist keras
from keras.utils import np_utils

 #load Mnist from server
(mnist_train, mnist_test), mnist_data = tfds.load("mnist", split=['train', 'test'], as_supervised=True, shuffle_files=True, with_info=True,)
 
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
  
#test
mnist_test =mnist_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
mnist_test = mnist_test.batch(128)
mnist_test = mnist_test.cache()
mnist_test = mnist_test.prefetch(tf.data.AUTOTUNE)
print(mnist_data, mnist_train, mnist_test)
#(x_train, y_train),(x_test, y_test) = mnist.load_data() keras

# training data : 60000 samples
# reshape and normalize input data
#x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
#x_train = x_train.astype('float32')
#x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
#x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
#x_test = x_test.astype('float32')
#x_test /= 255
#y_test = np_utils.to_categorical(y_test)

# Network
net = Network()
net.add(FcLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FcLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FcLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
net.fit(mnist_train, epochs=35, learning_rate=0.1)

# test on 3 samples
out = net.predict(mnist_test)
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(mnist_test)