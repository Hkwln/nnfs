import numpy as np

from fully_connected_layer import FcLayer
class Network:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.loss_prime = None
    #add layer to network
    def add(self,layer):
        self.layers.append(layer)
    #set loss to use
    def use(self, loss, loss_prime):
        self.loss =loss
        self.loss_prime = loss_prime
    #predict output for given input
    def predict(self,input_data):
        #sample dimension first
        samples = len(input_data)
        result = []
        #run network over all samples
        for i in range (samples):
            #forward propagation
            output = input_data[i]
            if output.ndim == 1:
                output = output.reshape(-1, 1)
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)  # Explicitly convert to NumPy array
        return result
    def get_weights(self):
        weights = []
        for layer in self.layers:
            if isinstance(layer, FcLayer):
                weights.append(layer.weights)
        return weights
    #train the network 
    def fit(self, x_train, y_train, epochs, learning_rate):
        # training loop
        for i in range(epochs):
            err = 0
            correct_predictions = 0
            train_losses = []
            train_accuracies = []
            for j in range(len(x_train)):
                # forward propagation
                output = x_train[j]
                if output.ndim == 1:
                    output = output.reshape(-1, 1)
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                # compute loss (only for displaying)
                err += self.loss(y_train[j], output)
                
                # compute accuracy
                if np.argmax(output) == np.argmax(y_train[j]):  # Explicitly convert to NumPy array
                    correct_predictions += 1
                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            
            # calculate average error and accuracy on all samples
            err /= len(x_train)
            accuracy = correct_predictions / len(x_train)
            train_losses.append(err)
            train_accuracies.append(accuracy)
    
            
            print('epoch %d/%d' % (i+1, epochs))
            print(f'Loss: {err}, Accuracy: {accuracy}')
    def save_weights(self, filename):
        weights = []
        for layer in self.layers:
            if isinstance(layer, FcLayer):
                weights.append({
                    'weights': layer.weights.tolist(),
                    'bias': layer.bias.tolist()
                })
        np.save(filename, weights)

    def load_weights(self, file_path):
        weights = np.load(file_path, allow_pickle=True)
        for layer, weight in zip(self.layers, weights):
            if isinstance(layer, FcLayer):
                layer.weights = np.array(weight['weights'])
                layer.bias = np.array(weight['bias'])
