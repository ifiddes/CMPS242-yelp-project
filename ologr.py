import numpy as np
from scipy.special import expit
from scipy import optimize

np.random.seed(6)

def h(z):
    z = np.clip(z, -500, 500)
    return np.log(1 + np.exp(z))

def g(z):
    z = np.clip(z, -500, 500)
    return expit(z)

class OrdinalLogisticRegressionAT(object):
    def __init__(self, lamb=1):
        """
        All-Threshold ordinal logistic regression for CMPS242
        """
        self.lamb = lamb

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

        def loss(x0, X, y):
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
            # final = total + (lamb / 2.) * np.asscalar(w.dot(w))
            return total + (self.lamb / 2.) * np.asscalar(w.dot(w))

        def grad(x0, X, y):
            """
            :param x0: initial parameters
            :param X: data matrix
            :param y: label vector
            :return: gradient array
            """
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
            w_grad = np.sum([w_grad.sum(axis=1), np.multiply(self.lamb, w).T], axis=0)

            ones = np.multiply(-1, np.ones(X.shape[0])).reshape(X.shape[0], 1)
            theta_grad = np.empty((1, l))
            for k in range(l):
                a = X.dot(w) - theta[k]
                b = np.multiply(s[k], a)
                c = np.multiply(s[k], g(b))
                theta_grad[:, k] = np.asscalar(ones.T.dot(c))
            return np.hstack((w_grad, theta_grad)).flatten()

        x0 = np.random.randn(X.shape[1] + classes.size - 1) / X.shape[1]
        # Initialize weights at zero
        x0[: X.shape[1]] = 0.
        # Sort and scale initial threshold values by the number of classes
        x0[X.shape[1]:] = np.sort(classes.size * np.random.rand(classes.size - 1))

        # print optimize.check_grad(loss, grad, x0, X, y)
        out = optimize.minimize(loss, x0, args=(X, y), jac=grad, method='BFGS')

        w, theta = np.split(out.x, [X.shape[1]])
        return w, theta

    def predict(self, w, theta, x):
        """
        :param w: weights
        :param theta: class thresholds
        :param x: vector
        :return: index of class
        """
        # Create theta vector assuming that 0 and l are - and + inf, respectively
        unique_theta = np.empty(len(theta) + 2)
        unique_theta[0] = -np.inf
        unique_theta[-1] = np.inf  # p(y <= max_level) = 1
        unique_theta[1: -1] = np.sort(np.unique(theta))
        out = x.dot(w)
        return np.argmax(out < unique_theta, axis=0) - 1


class OrdinalLogisticRegressionIT(object):
    def __init__(self, lamb=1):
        """
        Intermediate threshold ordinal logistic regression for CMPS242
        """
        self.lamb = lamb

    def train(self, X, y):
        """
        :param X: data matrix
        :param y: label vector
        :return: weight vector, threshold vector
        """
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

        def loss(x0, X, y):
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
                if label == 0:
                    total += h(row.dot(w) - theta[label])
                elif label == l:
                    total += h(theta[l-1] - row.dot(w))
                else:
                    total += h(theta[label - 1] - row.dot(w)) + h(row.dot(w) - theta[label])
            return total + (self.lamb / 2.) * np.asscalar(w.dot(w))

        def grad(x0, X, y):
            """
            :param x0: initial parameters
            :param X: data matrix
            :param y: label vector
            :return: gradient array
            """
            l = max(y)
            w, theta = np.split(x0, [X.shape[1]])
            w.shape = (X.shape[1], 1)

            theta_vec = [theta[label] if label != l else np.inf for label in y]
            theta_minus = []
            for label in y:
                if label == 0:
                    theta_minus.append(-np.inf)
                else:
                    theta_minus.append(theta[label - 1])

            theta_vec = np.array(theta_vec).reshape((X.shape[0], 1))
            theta_minus = np.array(theta_minus).reshape((X.shape[0], 1))

            a = g(X.dot(w) - theta_vec)
            b = g(theta_minus - X.dot(w))
            c = a - b
            d = X.T.dot(c)
            e = np.multiply(self.lamb, w)

            w_grad = d + e

            theta_grad = np.empty(l)
            for k in range(l):
                first = 0
                second = 0
                for row, label in zip(X, y):
                    if label - 1 == k:
                        first += g(theta[k] - row.dot(w))
                    elif label == k:
                        second += g(row.dot(w) - theta[k])
                    else:
                        pass
                theta_grad[k] = first - second

            return np.hstack((w_grad.reshape(X.shape[1]), theta_grad)).flatten()

        x0 = np.random.randn(X.shape[1] + classes.size - 1) / X.shape[1]
        # Initialize weights at zero
        x0[: X.shape[1]] = 0.
        # Sort and scale initial threshold values by the number of classes
        x0[X.shape[1]:] = np.sort(classes.size * np.random.rand(classes.size - 1))

        # print 'check grad'
        # print optimize.check_grad(loss, grad, x0, X, y)

        out = optimize.minimize(loss, x0, args=(X, y), jac=grad, method='BFGS')

        w, theta = np.split(out.x, [X.shape[1]])
        return w, theta

    def predict(self, w, theta, x):
        """
        :param w: weights
        :param theta: class thresholds
        :param x: vector
        :return: index of class
        """
        # Create theta vector assuming that 0 and l are - and + inf, respectively
        unique_theta = np.empty(len(theta) + 2)
        unique_theta[0] = -np.inf
        unique_theta[-1] = np.inf  # p(y <= max_level) = 1
        unique_theta[1: -1] = np.sort(np.unique(theta))
        out = x.dot(w)
        # print 'out: ', out
        # print unique_theta
        # print out < unique_theta
        # print np.argmax(out < unique_theta)
        return np.argmax(out < unique_theta, axis=0) - 1

if __name__ == '__main__':
    """
    Unit test
    """
    c = OrdinalLogisticRegressionIT(lamb=2)
    w, theta = c.train(np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]), np.array([1, 2, 3]))
    X = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]])
    y = np.array([1, 2, 3])
    for row, truth in zip(X, y):
        label = y[c.predict(w, theta, row)]
        assert label == truth, 'Model broke...'

