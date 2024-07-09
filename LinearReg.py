import random
import types
import pandas as pd
import numpy as np
from numpy import typing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score


class Metrics:
    def __init__(self, metric):
        self._metrics = {
            'mae': lambda y, y_pred: (y_pred - y).abs().mean(),
            'mse': lambda y, y_pred: ((y_pred - y) ** 2).mean(),
            'rmse': lambda y, y_pred: ((y_pred - y) ** 2).mean() ** 0.5,
            'mape': lambda y, y_pred: 100 * ((y - y_pred) / y).abs().mean(),
            'r2': lambda y, y_pred: 1 - (sum((y - y_pred) ** 2) / sum((y - y.mean()) ** 2))
        }
        self.metric = metric


class MyLineReg(Metrics):
    def __init__(self, n_iter: int = 100,
                 learning_rate: int | float | types.FunctionType = 0.1, 
                 weights: None | pd.DataFrame | np.ndarray | list = None,
                 metric: None | str = None,
                 reg: None | str = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0, 
                 sgd_sample: int | float = None, 
                 random_state: int = 42
                 ):
        super().__init__(metric)
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.__weights = weights
        self.__error = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.__random_state = random_state
        self.sgd_sample = sgd_sample

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series, verbose: bool | int = False):
        random.seed(self.__random_state)
        self.__check_reg_attrs()
        
        X.insert(0, 'Bias', np.ones(X.shape[0]))

        if not self.__weights:
            self.__weights = np.ones(X.shape[1])

        n = X.shape[0]

        if isinstance(self.sgd_sample, float):
            self.sgd_sample = int(self.sgd_sample * n)

        for i in range(self.n_iter):

            y_pred = X.dot(self.__weights).values

            error = self._metrics[self.metric](y, y_pred)

            if verbose and ((i + 1) % verbose == 0 or i == 0):
                print(f'{i + 1} | loss: {error} | {self.metric}' if i != 0 
                      else f'start | loss: {error} | {self.metric}')

            if self.sgd_sample:
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                gradient = ((2 / self.sgd_sample) *
                            (y_pred[sample_rows_idx] - y[sample_rows_idx]).dot(X.iloc[sample_rows_idx])
                            + self.l1_coef * np.sign(self.__weights) + 2 * self.l2_coef * self.__weights)
            else:
                gradient = (2 / n) * (y_pred - y).dot(X) + self.l1_coef * np.sign(self.__weights) + 2 * self.l2_coef * self.__weights
            
            self.__reducing_the_weight(i, gradient)
                
        if self.n_iter:
            y_pred = X.dot(self.__weights).values
            self.__error = self._metrics[self.metric](y, y_pred) if self._metrics else None

    def predict(self, X):
        X.insert(0, 'Bias', np.ones(X.shape[0]))

        return X.dot(self.__weights).values

    def get_coef(self):
        return self.__weights[1:]

    def get_best_score(self):
        return self.__error

    def __check_reg_attrs(self):
        if not self.reg:
            self.l1_coef, self.l2_coef = 0, 0
        elif self.reg == 'l1':
            self.l2_coef = 0
        elif self.reg == 'l2':
            self.l1_coef = 0
    
    def __reducing_the_weight(self, i, gradient):
        if isinstance(self.learning_rate, (int, float)):
            self.__weights -= self.learning_rate * gradient
        else:
            self.__weights -= self.learning_rate(i + 1) * gradient


cls = MyLineReg(metric='mape', reg='l2', l2_coef=0.123, sgd_sample=2)
sk_cls = LinearRegression()
ls = Lasso(alpha=0.123)

X = pd.DataFrame({
    '_X1': [1, 2, 3, 4],
})

X_test = pd.DataFrame({
    '_X1': [2, 3, 4, 5]
})

y = pd.Series([2.3, 3.2, 4.5, 5.7])

cls.fit(X, y, verbose=50)
ls.fit(X, y)
sk_cls.fit(X, y)

print('MyLinPred', cls.predict(X_test))
print('sklearn:', sk_cls.predict(X_test))
print('Lasso:', ls.predict(X_test))
print()
print('MyLinPred', cls.get_best_score())
print('sklearn:', r2_score(y, sk_cls.predict(X)))
print('Lasso:', r2_score(y, ls.predict(X)))
