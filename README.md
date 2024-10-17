# nnfs
neural network from scratch, at least I tried
using tensorflow dataset
 latest error:
shapes (128,28,28,1) and (784,100) not aligned: 1 (dim 3) != 784 (dim 0)
  File "*/GitHub/nnfs/fully_connected_layer.py", line 14, in forward_propagation
    self.output = np.dot(self.input, self.weights) + self.bias
  File "*/GitHub/nnfs/network.py", line 37, in fit
    output = layer.forward_propagation(output)
  File "*/GitHub/nnfs/mnist_nn_fc.py", line 81, in <module>
    net.fit(x_train, y_train , epochs=35, learning_rate=0.1)