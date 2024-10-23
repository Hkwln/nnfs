from baselayerclass import Layer
import numpy as np

#inherit from base class Layer
class FcLayer(Layer):
    #input_size = number of input neurons
    #output_size = number of output neurons=
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.weights = np.random.rand(input_size,output_size) -0.5
        self.bias = np.random.rand(1,output_size) -0.5
       
    #returns outpot for a given input
    def forward_propagation(self, input_data):
        #ensure input_data is 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    #computes dE/dW, dE/dB for a fiven output_error = dE/dY. REturns input_errors=dE/dX.
    def backward_propagation(self,output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        #dBias = output_error

        #update paramaters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
