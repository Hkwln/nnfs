# nnfs
neural network from scratch, at least I tried
using tensorflow dataset
 latest error:
Only integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid indices, got array([[-0.99879159,  0.61818112,  0.91265485, -0.88631204, -0.42409893,
         0.08888091, -0.19478377, -0.02089696, -0.91586014, -0.96646877]])
  File "C:\Users\spaul\neural network from scratch\nnfs\network.py", line 45, in fit
    err += self.loss(y_train[j, output])
  File "C:\Users\spaul\neural network from scratch\nnfs\mnist_nn_fc.py", line 77, in <module>
    net.fit(images, labels , epochs=35, learning_rate=0.1)
TypeError: Only integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid indices, got array([[-0.99879159,  0.61818112,  0.91265485, -0.88631204, -0.42409893,
         0.08888091, -0.19478377, -0.02089696, -0.91586014, -0.96646877]])