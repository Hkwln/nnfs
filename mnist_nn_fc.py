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
#print(mnist_data, mnist_train, mnist_test)

# Network
net = Network()
net.add(FcLayer(784, 128))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100) copilot says 128
net.add(ActivationLayer(tanh, tanh_prime)) # or relu and relu_prime
net.add(FcLayer(128, 64))                   # input_shape=(1, 100)      ;   output_shape=(1, 50) copilot says 64

net.add(ActivationLayer(tanh, tanh_prime))
net.add(FcLayer(64, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...

#trying out copilots suggestion
for images, labels in mnist_train.take(1):
    images = tf.reshape(images, (images.shape[0], -1)) #flatten the images
    labels = tf.one_hot(labels, depth =10) #one hot encoding the labels
    images, labels = images.numpy(), labels.numpy()
    out = net.predict(images)
net.use(mse, mse_prime)
net.fit(images, labels , epochs=50, learning_rate=0.001)
#net.loss = net.evaluate(images, labels)


# test on 3 samples 
#out = net.predict(images[:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(labels[:3], end="\n")

#trying to visualize training process doesn't work yet
#loss_history = net.loss
#plt.plot(loss_history)
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.title('Training Loss over Epochs')
#plt.show()