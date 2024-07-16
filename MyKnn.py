import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MyKNNClf:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.__k = k
        self.train_size = None
        self.x_train = None
        self.y_train = None
        self._metric = metric
        self.__weight = weight

    def __str__(self):
        return f'MyKNNClf class: k={self.__k}'

    def fit(self, x_, y_):
        self.x_train, self.y_train = x_.values, y_.values
        self.train_size = x_.shape

    def predict_proba(self, x_test):
        x_test = x_test.values
        result = []
        result_h = []
        for x in x_test:
            distances = self.__choose_metrics(x)
            ind = np.argsort(distances)[:self.__k]
            if self.__weight == 'uniform':
                result_h.append((self.y_train[ind] == 1).sum() / len(self.y_train[ind]))
            else:
                result_h.append(self._choose_weight(x, result))

        return np.array(result_h)

    def predict(self, x_test):
        x_test = x_test.values
        result = []
        for x in x_test:
            self._choose_weight(x, result)

        return np.array(result)

    @staticmethod
    def _calculate(index_ones, index_zeros, result, dist_or_pos):
        q_1 = np.sum(1 / index_ones) / np.sum(1 / dist_or_pos)
        q_0 = np.sum(1 / index_zeros) / np.sum(1 / dist_or_pos)
        result.append(int(q_1 > q_0))
        return q_1

    def _choose_weight(self, x, result):
        distances = self.__choose_metrics(x)
        ind = np.argsort(distances)[:self.__k]

        if self.__weight == 'uniform':
            result.append(1) if self.y_train[ind].sum() >= (self.y_train[ind] == 0).sum() else result.append(0)
        elif self.__weight == 'rank':
            index_ones = np.where(self.y_train[ind] == 1)[0] + 1
            index_zeros = np.where(self.y_train[ind] == 0)[0] + 1
            return self._calculate(index_ones, index_zeros, result, np.arange(1, len(self.y_train[ind]) + 1))
        elif self.__weight == 'distance':
            index_ones = distances[ind][np.where(self.y_train[ind] == 1)[0]]
            index_zeros = distances[ind][np.where(self.y_train[ind] == 0)[0]]
            return self._calculate(index_ones, index_zeros, result, distances[ind])

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


X, y = make_classification(n_samples=500, n_features=20, n_informative=2, n_redundant=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
y_train, y_test = pd.Series(y_train), pd.Series(y_test)

my_cls = MyKNNClf(metric='cosine', weight='rank')
my_cls.fit(X_train, y_train)
print(my_cls.predict(X_test))
print(my_cls.predict_proba(X_test))
print(accuracy_score(y_test, my_cls.predict(X_test)))
