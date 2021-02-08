import math
import numpy as np


class Regression(object):
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """ 
        Initialzie weights randomly [-1/N, 1/N]
        """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.training_error = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        pass

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred
        
        

