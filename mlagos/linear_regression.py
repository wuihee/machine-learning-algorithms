import numpy as np


class LinearRegression:
    """
    Linear regression model.

    Attributes:
        lr (float): Model's learning rate (alpha).
        n_iters (int): The number of iteration the model updates for gradient descent.
        weights (np.ndarray): The coefficients for each feature.
        bias (float): The constant added at the end of the model's prediction.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initializes linear regression model with a learning rate and number of iterations.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains the model given input features and target values.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples).
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Apply gradient descent.
        for _ in range(self.n_iters):
            y_predict = np.dot(X, self.weights) + self.bias

            # Calculate derivatives.
            dw = (1 / num_samples) * np.dot(X.T, (y_predict - y))
            db = (1 / num_samples) * np.sum(y_predict - y)

            # Update model parameters.
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predicts the target values.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values of shape (n_samples).
        """
        return np.dot(X, self.weights) + self.bias
