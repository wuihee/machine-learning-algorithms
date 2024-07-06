import numpy as np


class KMeans:
    def __init__(self, n_clusters=5, n_iters=100):
        self.n_clusters = n_clusters
        self.n_iters = n_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.n_iters):
            self.labels = self._assign_clusters(X)
            self.centroids = self._calculate_centroids(X)
        return self

    def predict(self, X):
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        indices = np.zeros(X.shape[0])
        for i in range(len(X)):
            indices[i] = np.argmin([np.linalg.norm(X[i] - mu) for mu in self.centroids])
        return indices

    def _calculate_centroids(self, X):
        return np.array(
            [X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)]
        )
