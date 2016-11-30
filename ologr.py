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

            grad_vec = np.zeros(X.shape[1] + l)
            for j in range(X.shape[1]):
                total = 0
                for row, label in zip(X, y):
                    row = np.array(row)
                    first = 0
                    for k in range(label, l):
                        first += row[j] * g(row.dot(w) - theta[k])
                    second = 0
                    for k in range(label):
                        second += row[j] * g(theta[k] - row.dot(w))
                    total += first - second + lamb * w[j]
                grad_vec[j] = total

            for k, j in zip(range(len(theta)), range(X.shape[1], X.shape[1] + l)):
                print k
                s = np.array([1 if k > label else -1 for label in y])
                print s







        x0 = np.random.randn(X.shape[1] + classes.size - 1) / X.shape[1]
        # Initialize weights at zero
        x0[: X.shape[1]] = 0.
        # Sort and scale initial threshold values by the number of classes
        x0[X.shape[1]:] = np.sort(classes.size * np.random.rand(classes.size - 1))

        print X
        print y
        print x0

        #print loss(x0, X, y)
        print grad(x0, X, y)


# # .. utility arrays used in f_grad ..
# alpha = 0.
# k1 = np.sum(y == classes[0])
# E0 = (y[:, np.newaxis] == np.unique(y)).astype(np.int)
# E1 = np.roll(E0, -1, axis=-1)
# E1[:, -1] = 0.
#
# # Make a sparse matrix
# E0, E1 = map(sparse.csr_matrix, (E0.T, E1.T))
#
#
# def f_obj(x0, X, y):
#     """
#     Objective function
#     """
#     w, theta_0 = np.split(x0, [X.shape[1]])
#     theta_1 = np.roll(theta_0, 1)
#     t0 = theta_0[y]
#     z = np.diff(theta_0)
#
#     Xw = X.dot(w)
#     a = t0 - Xw
#     b = t0[k1:] - X[k1:].dot(w)
#     c = (theta_1 - theta_0)[y][k1:]
#
#     if np.any(c > 0):
#         return 1e20
#
#     #loss = -(c[idx] + np.log(np.exp(-c[idx]) - 1)).sum()
#     loss = -np.log(1 - np.exp(c)).sum()
#
#     loss += b.sum() - np.log(sigmoid(b)).sum() \
#         + np.log(sigmoid(a)).sum() \
#         + .5 * alpha * w.dot(w) - np.log(z).sum()  # penalty
#     if np.isnan(loss):
#         pass
#         #import ipdb; ipdb.set_trace()
#     return loss
#
# def f_grad(x0, X, y):
#     """
#     Gradient of the objective function
#     """
#     w, theta_0 = np.split(x0, [X.shape[1]])
#     theta_1 = np.roll(theta_0, 1)
#     t0 = theta_0[y]
#     t1 = theta_1[y]
#     z = np.diff(theta_0)
#
#     Xw = X.dot(w)
#     a = t0 - Xw
#     b = t0[k1:] - X[k1:].dot(w)
#     c = (theta_1 - theta_0)[y][k1:]
#
#     # gradient for w
#     phi_a = sigmoid(a)
#     phi_b = sigmoid(b)
#     grad_w = -X[k1:].T.dot(phi_b) + X.T.dot(1 - phi_a) + alpha * w
#
#     # gradient for theta
#     idx = c > 0
#     tmp = np.empty_like(c)
#     tmp[idx] = 1. / (np.exp(-c[idx]) - 1)
#     tmp[~idx] = np.exp(c[~idx]) / (1 - np.exp(c[~idx]))  # should not need
#     grad_theta = (E1 - E0)[:, k1:].dot(tmp) \
#                  + E0[:, k1:].dot(phi_b) - E0.dot(1 - phi_a)
#
#     grad_theta[:-1] += 1. / np.diff(theta_0)
#     grad_theta[1:] -= 1. / np.diff(theta_0)
#     out = np.concatenate((grad_w, grad_theta))
#     return out
#
# x0 = np.random.randn(X.shape[1] + classes.size) / X.shape[1]
# x0[:X.shape[1]] = 0.
# x0[X.shape[1]:] = np.sort(classes.size * np.random.rand(classes.size))
#
# print optimize.check_grad(f_obj, f_grad, x0, X, y)
# print optimize.approx_fprime(x0, f_obj, 0.000001, X, y).sum()

if __name__ == '__main__':
    c = OrdinalLogisticRegression()
    c.train(np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]), np.array([1, 2, 3]))
