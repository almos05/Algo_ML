import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric='mse', reg=None, l1_coef: float = 0,
                 l2_coef: float = 0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.__error = None
        self.__metrics = {
            'mae': lambda y, y_pred: abs((y_pred - y)).mean(),
            'mse': lambda y, y_pred: ((y_pred - y) ** 2).mean(),
            'rmse': lambda y, y_pred: ((y_pred - y) ** 2).mean() ** 0.5,
            'mape': lambda y, y_pred: 100 * (abs((y - y_pred) / y).mean()),
            'r2': lambda y, y_pred: 1 - (sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2))
        }
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        if not self.reg:
            self.l1_coef, self.l2_coef = 0, 0
        elif self.reg == 'l1':
            self.l2_coef = 0
        elif self.reg == 'l2':
            self.l1_coef = 0

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X, y, verbose=False):

        X.insert(0, 'Bias', np.ones(X.shape[0]))

        self.weights = np.ones(X.shape[1])

        n = X.shape[0]

        for i in range(self.n_iter):

            y_pred = X.dot(self.weights).values

            error = self.__metrics[self.metric](y, y_pred)

            if verbose and ((i + 1) % verbose == 0 or i == 0):
                print(
                    f'{i + 1} | loss: {error} | {self.metric}' if i != 0 else f'start | loss: {error} | {self.metric}')

            gradient_loss = (2 / n) * (y_pred - y).dot(X)

            self.weights -= self.learning_rate * (gradient_loss + self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights)

        if self.n_iter:
            y_pred = X.dot(self.weights).values

            self.__error = self.__metrics[self.metric](y, y_pred)

    def predict(self, X):
        X.insert(0, 'Bias', np.ones(X.shape[0]))

        return X.dot(self.weights).values

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.__error


cls = MyLineReg(metric='r2', reg='l2', l2_coef=0.123)
sk_cls = LinearRegression()
ls = Lasso(alpha=0.123)

x = pd.DataFrame({
    '_X1': [1, 2, 3, 4],
})

x_test = pd.DataFrame({
    '_X1': [2, 3, 4, 5]
})

y = pd.Series([2.3, 3.2, 4.5, 5.7])

cls.fit(x, y, verbose=50)
ls.fit(x, y)
sk_cls.fit(x, y)

print('MyLinPred', cls.predict(x_test))
print('sklearn:', sk_cls.predict(x_test))
print('Lasso:', ls.predict(x_test))
print()
print('MyLinPred', cls.get_best_score())
print('sklearn:', r2_score(y, sk_cls.predict(x)))
print('Lasso:', r2_score(y, ls.predict(x)))
