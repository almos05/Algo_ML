import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from scipy.special import expit


class Metrics:
    def __init__(self):
        self._metrics = {
            'accuracy': lambda y, y_pred: (y == y_pred).mean(),
            'precision': self.__precision,
            'recall': self.__recall,
            'f1': self.__f1,
            'roc_auc': self.__roc_auc
        }
        self._score = None

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
        return 2 * precision * recall / (recall + precision)

    @staticmethod
    def __roc_auc(y, y_proba):
        sorted_indices = np.argsort(y_proba)[::-1]
        y_sorted = y[sorted_indices]

        tpr = np.cumsum(y_sorted) / y_sorted.sum()
        fpr = np.cumsum(1 - y_sorted) / (len(y_sorted) - y_sorted.sum())

        return np.trapz(tpr, fpr)


class MyLogReg(Metrics):
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate=0.1,
                 metric='accuracy',
                 reg=None,
                 l1_coef=0.1,
                 l2_coef=0.1,
                 sgd_sample=None,
                 random_state=42
                 ):
        super().__init__()
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self.__weights = None
        self.metric = metric
        self._reg = reg
        self._l1_coef = l1_coef
        self._l2_coef = l2_coef
        self._sgd_sample = sgd_sample
        self.__random_state = random_state

    def __str__(self):
        return f'MyLogReg class: n_iter={self._n_iter}, learning_rate={self._learning_rate}'

    def fit(self, X, y, verbose=False):
        random.seed(self.__random_state)
        X = X.copy()
        y = y.copy()
        n = X.shape[0]

        if isinstance(self._sgd_sample, float):
            self._sgd_sample = int(self._sgd_sample * n)
        self.__check_reg_attrs()

        X.insert(0, 'Bias', np.ones(n))

        self.__weights = np.ones(X.shape[1])

        for i in range(self._n_iter):
            z = np.dot(X, self.__weights)
            y_pred = expit(z)

            if self._sgd_sample:
                sample_rows_idx = random.sample(range(X.shape[0]), self._sgd_sample)
                gradient = 1 / self._sgd_sample * (y_pred[sample_rows_idx] - y[sample_rows_idx]).dot(
                    X.iloc[sample_rows_idx])
            else:
                gradient = 1 / n * (y_pred - y).dot(X)

            gradient += self._l1_coef * np.sign(self.__weights) + self._l2_coef * 2 * self.__weights

            self.__reducing_the_weight(i, gradient)

            if self.metric == 'roc_auc':
                self._score = self._metrics[self.metric](y, self.predict_proba(X.iloc[:, 1:]))
            else:
                self._score = self._metrics.get(self.metric, None)(y, self.predict(X.iloc[:, 1:]))

            epsilon = 1e-15
            log_loss = (- 1) * (y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

            if verbose and i % verbose == 0:
                print(f'{i + 1} | {log_loss.mean()} | {self._score}')

    def predict_proba(self, X):
        X = X.copy()
        X.insert(0, 'Bias', np.ones(X.shape[0]))

        z = np.dot(X, self.__weights)

        return expit(z)

    def get_coef(self):
        return self.__weights[1:]

    def predict(self, X):
        temp = self.predict_proba(X)

        return (temp > 0.5).astype(int)

    def get_best_score(self):
        return self._score

    def __check_reg_attrs(self):
        if not self._reg:
            self._l1_coef, self._l2_coef = 0, 0
        elif self._reg == 'l1':
            self._l2_coef = 0
        elif self._reg == 'l2':
            self._l1_coef = 0

    def __reducing_the_weight(self, i, gradient):
        if isinstance(self._learning_rate, (int, float)):
            self.__weights -= self._learning_rate * gradient
        else:
            self.__weights -= self._learning_rate(i + 1) * gradient


X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)

sk_cls = LogisticRegression(max_iter=1000)
sk_cls.fit(X_train, y_train)
y_pred_sk = sk_cls.predict(X_train)

my_cls = MyLogReg(n_iter=1000, metric='accuracy', reg='elastic', learning_rate=lambda iter: 0.5 * (0.85 ** iter), sgd_sample=0.1)
my_cls.fit(X_train, y_train, verbose=20)
y_pred = my_cls.predict(X_train)

print('accuracy:', accuracy_score(y_train, y_pred_sk))
print(my_cls.get_best_score())

print('\nprecision:', precision_score(y_train, y_pred_sk))
print(my_cls.get_best_score())

print('\nrecall:', recall_score(y_train, y_pred_sk))
print(my_cls.get_best_score())

print('\nf1:', f1_score(y_train, y_pred_sk))
print(my_cls.get_best_score())

print('\nroc_auc:', roc_auc_score(y_train, y_pred_sk))
print(my_cls.get_best_score())