from baselayerclass import Layer
import numpy as np

#inherit from base class Layer
class FcLayer(Layer):
    #input_size = number of input neurons
    #output_size = number of output neurons=
    def __init__(self, input_size, output_size, beta1=0.9, beta2 = 0.999, epsilon = 1e-8) -> None:
        super().__init__()
        
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        #self.beta1 = beta1
        #self.beta2 = beta2
        #self.epsilon = epsilon
       
    #returns outpot for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        if self.input.ndim == 1:
            self.input = self.input.reshape(-1, 1)
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output
    #now i am trying to train the network using a mini-batch stochastic gradient descent
    #The training data is a list of tubles with(image, label)
    
    def SGD(self, input_data, epochs, mini_batch_size, learning_rate, test_data=None):
        n = len(input_data)
        for j in range(epochs):
            #time1 = time.time()
            np.random.shuffle(input_data)
            mini_batches = [input_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            #time2 = time.time()
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    #now we update the networks weights and biases
    #mini batch is a list of tuples (x,y)
    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.bias = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.bias, nabla_b)]

    #computes dE/dW, dE/dB for a fiven output_error = dE/dY. REturns input_errors=dE/dX.
    #now we need to reuturn the tuple nabla_b, nabla w
    #they are representing the gradient for the cost function C_x 
    #nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self.bias and self.weights
    def backward_propagation(self, output_error, learning_rate):
        if output_error.ndim == 1:
            output_error = output_error.reshape(-1, 1)
        input_error = np.dot(self.weights.T, output_error)
        weights_error = np.dot(output_error, self.input.T)
        
        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=1, keepdims=True)
        
        return input_error

    #old code
    # def backward_propagation(self, output_error, learning_rate):
    #     input_error = np.dot(output_error, self.weights.T)
    #     weights_error = np.dot(self.input.T, output_error)
    #     self.weights -= learning_rate * weights_error
    #     self.bias -= learning_rate * output_error
    #     return input_error

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward_propagation(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    #return the vector of the cost function derivative
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


    #sigmoid activation function
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

