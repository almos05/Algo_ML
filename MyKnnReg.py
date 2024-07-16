import numpy as np
import pandas as pd


class MyKNNReg:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.__k = k
        self.x_train = None
        self.y_train = None
        self.train_size = None
        self._metric = metric
        self._weight = weight

    def __str__(self):
        return f'MyKNNReg class: k={self.__k}'

    def fit(self, x_train, y_train):
        self.x_train, self.y_train = x_train, y_train
        self.train_size = x_train.shape

    def predict(self, x_test):
        x_test = x_test.values

        result = []
        for x in x_test:
            distances = self.__choose_metrics(x)
            ind = np.argsort(distances)[:self.__k]

            if self._weight == 'uniform':
                pred = self.y_train[ind].mean()
            elif self._weight == 'rank':
                weights = 1 / (np.arange(1, self.__k + 1))
                weights /= weights.sum()
                pred = weights * self.y_train[ind]
            elif self._weight == 'distance':
                weights = 1 / distances[ind]
                weights /= weights.sum()
                pred = weights * self.y_train[ind]
            else:
                raise ValueError("Unknown weight type")

            result.append(pred)

        return np.array(result)

    def _euclidean_distances(self, x):
        return np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))

    def _chebyshev_distances(self, x):
        return np.max(np.abs(self.x_train - x), axis=1)

    def _manhattan_distances(self, x):
        return np.sum(np.abs(self.x_train - x), axis=1)

    def _cosine_distances(self, x):
        return (1 - np.sum(self.x_train * x, axis=1) /
                (np.sqrt(np.sum(self.x_train ** 2, axis=1)) * np.sqrt(np.sum(x ** 2))))

    def __choose_metrics(self, x):
        if self._metric == 'euclidean':
            return self._euclidean_distances(x)
        elif self._metric == 'chebyshev':
            return self._chebyshev_distances(x)
        elif self._metric == 'manhattan':
            return self._manhattan_distances(x)
        elif self._metric == 'cosine':
            return self._cosine_distances(x)
        return None