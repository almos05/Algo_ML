import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MyKNNClf:
    def __init__(self, k=3, metric='euclidean'):
        self.__k = k
        self.train_size = None
        self.x_train = None
        self.y_train = None
        self._metric = metric

    def __str__(self):
        return f'MyKNNClf class: k={self.__k}'

    def fit(self, x_, y_):
        self.x_train, self.y_train = x_.values, y_.values
        self.train_size = x_.shape

    def predict_proba(self, x_test):
        x_test = x_test.values
        result = []
        for x in x_test:
            distances = self.__choose_metrics(x)
            ind = np.argsort(distances)[:self.__k]
            result.append((self.y_train[ind] == 1).sum() / len(self.y_train[ind]))

        return np.array(result)

    def predict(self, x_test):
        x_test = x_test.values
        result = []
        for x in x_test:
            distances = self.__choose_metrics(x)
            ind = np.argsort(distances)[:self.__k]
            if y_train[ind].sum() >= (y_train[ind] == 0).sum():
                result.append(1)
            else:
                result.append(0)

        return np.array(result)

    def _euclidean_distances(self, x):
        return np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))

    def _chebyshev_distances(self, x):
        return np.max(np.abs(self.x_train - x), axis=1)

    def _manhattan_distances(self, x):
        return np.sum(np.abs(self.x_train - x), axis=1)

    def _cosine_distances(self, x):
        return 1 - np.sum(self.x_train * x, axis=1) / (np.sqrt(np.sum(self.x_train ** 2, axis=1)) * np.sqrt(np.sum(x ** 2)))

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

my_cls = MyKNNClf(metric='cosine')
my_cls.fit(X_train, y_train)
print(my_cls.predict(X_test))
print(my_cls.predict_proba(X_test))
print(accuracy_score(y_test, my_cls.predict(X_test)))
