import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

    def get_coef(self):
        return self.__weights[1:]


x_train = pd.DataFrame({'X1': [300, 320, 450, 120, 700, 100]})
y_train = pd.Series([1, 0, 0, 1, 0, 1])

x_test = pd.DataFrame({'X1': [1000, 290, 430, 270, 310, 200]})

sk_cls = LogisticRegression(max_iter=100)
sk_cls.fit(x_train, y_train)
sk_cls.predict(x_train)

my_cls = MyLogReg()
my_cls.fit(x_train, y_train, verbose=3)

print('sk_cls:', sk_cls)
