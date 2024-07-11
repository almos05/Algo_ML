import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.special import expit


class Metrics:
    def __init__(self):
        self._metrics = {
            'accuracy': lambda y, y_pred: (y == y_pred).mean(),
            'precision': self.__precision,
            'recall': self.__recall,
            'f1': self.__f1
        }

    @staticmethod
    def __precision(y, y_pred):
        tp = ((y == 1) & (y_pred == 1)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        return tp / (tp + fp)

    @staticmethod
    def __recall(y, y_pred):
        tp = ((y == 1) & (y_pred == 1)).sum()
        fn = ((y == 1) & (y_pred == 0)).sum()
        return tp / (tp + fn)

    def __f1(self, y, y_pred):
        precision = self.__precision(y, y_pred)
        recall = self.__recall(y, y_pred)
        return None  # Change here


class MyLogReg(Metrics):
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate=0.1,
                 metric=None
                 ):
        super().__init__()
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self.__weights = None
        self.__metric = metric

    def __str__(self):
        return f'MyLogReg class: n_iter={self._n_iter}, learning_rate={self._learning_rate}'

    def fit(self, X, y, verbose=False):
        X = X.copy()
        y = y.copy()
        n = X.shape[0]

        X.insert(0, 'Bias', np.ones(n))

        self.__weights = np.ones(X.shape[1])

        for i in range(self._n_iter):
            z = np.dot(X, self.__weights)
            y_pred = expit(z)

            gradient = 1 / n * (y_pred - y).dot(X)

            self.__weights -= self._learning_rate * gradient

            epsilon = 1e-15
            log_loss = (- 1) * (y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

            if verbose and i % verbose == 0:
                print(f'{i + 1} | {log_loss.mean()}')

    def predict_proba(self, X):
        X = X.copy()
        X.insert(0, 'Bias', np.ones(X.shape[0]))

        z = np.dot(X, self.__weights)
        y_pred = expit(z)

        return y_pred

    def get_coef(self):
        return self.__weights[1:]

    def predict(self, X):
        temp = self.predict_proba(X)
        return (temp > 0.5).astype(int)

    def get_best_score(self, y, y_pred):
        return self._metrics[self.__metric](y, y_pred)


X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)

sk_cls = LogisticRegression(max_iter=100)
sk_cls.fit(X_train, y_train)
sk_cls.predict(X_train)

my_cls = MyLogReg(n_iter=100, metric='recall')
my_cls.fit(X_train, y_train, verbose=3)
y_pred = my_cls.predict(X_train)

print('accuracy:', accuracy_score(y_train, y_pred))
print(my_cls.get_best_score(y_train, y_pred))

print('\npresicion:', precision_score(y_train, y_pred))
print(my_cls.get_best_score(y_train, y_pred))

print('\nrecall:', recall_score(y_train, y_pred))
print(my_cls.get_best_score(y_train, y_pred))