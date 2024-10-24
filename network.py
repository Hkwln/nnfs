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
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
    def get_weights(self):
        weights = []
        for layer in self.layers:
            if isinstance(layer, FcLayer):
                weights.append(layer.weights)
        return weights
    #train the network 
    def fit(self,x_train, y_train, epochs, learning_rate):
        #training loop
        for i in range(epochs):
            err = 0
            for j in range(len(x_train)):
                #forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                #compute loss (only for displaying)
                err += self.loss(y_train[j], output)
                #Backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    
                    error = layer.backward_propagation(error, learning_rate)

            #calculate average error on all samples
            err /= len(x_train)
            print('epoch %d/%d error =%f' % (i+1, epochs, err))