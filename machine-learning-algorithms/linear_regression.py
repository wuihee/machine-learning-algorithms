import numpy as np


class LinearRegression:
    """
    Linear Regression model.

    Attributes:
        lr (float): Learning rate for gradient descent.
        n_iters (int): Number of iterations for gradient descent.
        weights (numpy.ndarray): Weights of the model.
        bias (float): Bias term of the model.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initializes the LinearRegression model with given learning rate and number of iterations.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.001.
            n_iters (int, optional): Number of iterations. Defaults to 1000.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the LinearRegression model using the given training data.

        Args:
            X (numpy.ndarray): Training data of shape (n_samples, n_features).
            y (numpy.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Apply gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute derivatives
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray):
        """
        Predicts the target values using the trained LinearRegression model.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Predicted target values of shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias
