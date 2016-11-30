import numpy as np
from scipy.special import expit
from scipy import optimize, sparse

np.random.seed(23)

def sigmoid(t):
    t = np.clip(t, -500, 500)
    return expit(t)

def h(z):
    z = np.clip(z, -500, 500)
    return np.log(1 + np.exp(z))

def g(z):
    z = np.clip(z, -500, 500)
    return expit(z)

class OrdinalLogisticRegression(object):
    def __init__(self):
        """
        Ordinal logistic regression for CMPS242
        """
        pass

    def train(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        idx = np.argsort(y)
        X = X[idx]
        y = y[idx]

        classes = np.unique(y)

        # Relabel classes starting at zero
        for i, cls in enumerate(classes):
            y[y == cls] = i
        classes = np.unique(y)

        k1 = np.sum(y == classes[0])

        def loss(x0, X, y, lamb=1):
            """
            :param x0: array containing w and theta values
            :param X: data
            :param y: labels
            :return: loss
            """
            l = max(y)
            w, theta = np.split(x0, [X.shape[1]])
            total = 0
            for row, label in zip(X, y):
                row = np.array(row)
                first = 0
                for k in range(label):
                    first += h(theta[k] - row.dot(w))
                second = 0
                for k in range(label, l):
                    second += h(row.dot(w) - theta[k])
                total += first + second
            return total + (lamb / 2.) * np.asscalar(w.dot(w))

        def grad(x0, X, y, lamb=1):
            l = max(y)
            w, theta = np.split(x0, [X.shape[1]])
            w.shape = (X.shape[1], 1)

            s = []
            for k in range(l):
                s_k = np.zeros(len(y))
                for i, label in enumerate(y):
                    s_k[i] = 1 if k >= label else -1
                s.append(s_k.reshape((len(y), 1)))

            w_grad = np.empty((X.shape[1], l))
            for k in range(l):
                a = X.dot(w) - theta[k]
                b = np.multiply(s[k], a)
                c = np.multiply(s[k], g(b))
                d = X.T.dot(c)
                w_grad[:, k] = d.T
            w_grad = np.sum([w_grad.sum(axis=1), np.multiply(lamb, w).T], axis=0)

            ones = np.multiply(-1, np.ones(X.shape[0])).reshape(X.shape[0], 1)
            theta_grad = np.empty((1, l))
            for k in range(l):
                a = X.dot(w) - theta[k]
                b = np.multiply(s[k], a)
                c = np.multiply(s[k], g(b))
                theta_grad[:, k] = np.asscalar(ones.T.dot(c))

            return np.hstack((w_grad, theta_grad))

        x0 = np.random.randn(X.shape[1] + classes.size - 1) / X.shape[1]
        # Initialize weights at zero
        x0[: X.shape[1]] = 0.
        # Sort and scale initial threshold values by the number of classes
        x0[X.shape[1]:] = np.sort(classes.size * np.random.rand(classes.size - 1))



if __name__ == '__main__':
    c = OrdinalLogisticRegression()
    c.train(np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]), np.array([1, 2, 3]))