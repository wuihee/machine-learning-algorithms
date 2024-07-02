import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000, threshold=0.5):
        self.lr = lr
        self.n_iters = n_iters
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return [0 if y <= self.threshold else 1 for y in y_pred]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
