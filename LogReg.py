import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.special import expit


class MyLogReg:
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate=0.1
                 ):
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self.__weights = None

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


X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)

sk_cls = LogisticRegression(max_iter=100)
sk_cls.fit(X_train, y_train)
sk_cls.predict(X_train)

my_cls = MyLogReg(n_iter=100)
my_cls.fit(X_train, y_train, verbose=3)
print(my_cls.predict_proba(X_train))
print(my_cls.predict(X_train).mea)