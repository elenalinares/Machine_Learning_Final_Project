#This file is a Neural Network coded only with numpy

#our beloved library:
import numpy as np

class NeuralNetworkRegressorScratch:
    def __init__(self, n_hidden = (16), learning_rate = 0.01, n_epochs = 100, random_state = None):
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
        y = np.asarray(y, dtype = float).reshape(-1, 1)

        n_samples, n_features = X.shape

        #initialize the weights 
        self.W1 = np.random.randn(n_features, self.n_hidden) * 0.01
        self.b1 = np.zeros((1, self.n_hidden))
        self.W2 = np.random.randn(self.n_hidden, 1) * 0.01
        self.b2 = np.zeros((1,1))

    #training loop
        for epoch in range(self.n_epochs):
            #--- Forward pass
            Z1 = X.dot(self.W1) + self.b1
            A1 = self._relu(Z1)       
            Z2 = A1.dot(self.W2) + self.b2
            y_pred = Z2 #linear output for regression

            #--- MSE
            
            diff = y - y_pred
            loss = np.mean(diff ** 2)
            loss = np.mean((y - y_pred) **2)

            #--- gradients 
            dZ2 = 2 * (y_pred - y) / n_samples
            dW2 = A1.T.dot(dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = dZ2.dot(self.W2.T)
            dZ1 = dA1 * self._relu_deriv(Z1)
            dW1 = X.T.dot(dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            #--- update the wights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            #--- print the progress made every 10 iterations
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

        return self
    

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Z1 = X.dot(self.W1) + self.b1
        A1 = self._relu(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        return Z2.ravel()

