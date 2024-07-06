import numpy as np


class KMeans:
    def __init__(self, n_clusters=5, n_iters=100):
        """
        Initialize K-Means.

        Args:
            n_clusters (int, optional): The number of clusters to search for.
            n_iters (int, optional): The number of iterations to converge.
        """
        self.n_clusters = n_clusters
        self.n_iters = n_iters

    def fit(self, X):
        """
        Repeat the optimization steps of assigning points to nearest clusters,
        and then moving clusters to the mean position of their points.

        Args:
            X (np.ndarray): Data points.

        Returns:
            self: Returns instance of self.
        """
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.n_iters):
            self.labels = self._assign_clusters(X)
            self.centroids = self._calculate_centroids(X)
        return self

    def predict(self, X):
        """
        Returns indices, where indices[i] is the cluster associated with index i.
        """
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        """
        Assign each point in X to its closest cluster.

        Args:
            X (np.ndarray): Data points.

        Returns:
            np.ndarray: NumPy array where indices[i] is the cluster associated
                        with the ith index.
        """
        indices = np.zeros(X.shape[0], dtype=int)
        for i in range(len(X)):
            indices[i] = np.argmin([np.linalg.norm(X[i] - mu) for mu in self.centroids])
        return indices

    def _calculate_centroids(self, X):
        """
        Calculate new positions of centroids.

        Args:
            X (np.ndarray): Data points.

        Returns:
            np.ndarray: The new positions of the centroids.
        """
        return np.array(
            [X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)]
        )
