#This file is a Neural Network coded only with numpy

#our beloved library:
import numpy as np

class NeuralNetworkRegressorScratch:
    def __init__(self, n_hidden = 10, learning_rate = 0.01, n_epochs = 100, random_state = None):
        """
        This is a simple 1-hidden-layer neural network regressor.

        n_hidden: number of neurons in the hidden layer
        learning_rate: step size for gradient descennt
        n_epochs: number of training iterations
        """

        if random_state is not None:
            np.random.seed(random_state)
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs



    def _relu(self, z):
        return np.maximum(0, z)
    

    def _relu_deriv(self, z):
        return (z > 0).astype(float)
    

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float)

        n_samples, n_features = X.shape

        #initialize the weights 
        self.W1 = np.random.randn(n_features, self.n_hidden) * 0.01
        self.b1 = np.zeros((1, self.n_hidden))
        self.W2 = np.random.randn(self.n_hidden, 1) * 0.01
        self.b2 = np.zeros((1,1))       