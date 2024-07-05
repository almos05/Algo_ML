import pandas as pd
import numpy as np


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, X, y, verbose=False):
        X.insert(0, 'Bias', np.ones(X.shape[0]))

        self.weights = np.ones(X.shape[1])

        n = X.shape[0]

        for i in range(self.n_iter):
            y_pred = X.dot(self.weights).values

            MSE = ((y_pred - y) ** 2).mean()

            if verbose and ((i + 1) % verbose == 0 or i == 0):
                print(f'{i + 1} | loss: {MSE}' if i != 0 else f'start | loss: {MSE}')

            gradient = (2 / n) * (y_pred - y).dot(X)

            self.weights -= self.learning_rate * gradient

    def get_coef(self):
        return self.weights[1:]


cls = MyLineReg()

x = pd.DataFrame({
    '_X1': [1, 2, 3, 4],
})

y = [2.3, 3.2, 4.5, 5.7]

cls.fit(x, y, verbose=50)
