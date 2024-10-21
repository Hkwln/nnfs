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
#test
mnist_test = mnist_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
mnist_test = mnist_test.batch(128)
mnist_test = mnist_test.cache()
mnist_test = mnist_test.prefetch(tf.data.AUTOTUNE)
print(mnist_data, mnist_train, mnist_test)

#hier kommt der image-label extract von den mnist training data (kopiert, muss noch getestet werden)

## Extract the images and labels
x_train, y_train = [], []
for image, label in tfds.as_numpy(mnist_train):
    x_train.append(image)
    y_train.append(label)

# Convert lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = x_train / 255.0

# Network
net = Network()
net.add(FcLayer(784, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FcLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FcLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...

net.use(mse, mse_prime)
net.fit(x_train, y_train , epochs=35, learning_rate=0.1)

# test on 3 samples
out = net.predict(mnist_test)
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(mnist_test)
