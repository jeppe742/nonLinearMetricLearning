import numpy as np
from scipy.special import softmax


def chi2_distance(x, y):
    """
    Calculates the chi2 distance between two multivariate inputs

    Args:
        x: [n', d] matrix, where the number of datapoints n' either has to be n or 1, and d is the dimension 
        y: [n, d] matrix, where n is the number of datapoints 
    """
    n, d = y.shape

    diff = x-y
    add = x+y

    # The first dot calculates (x-y)^2, but this will give a [n,n] matrix with all cross terms.
    # We are however only interested in the diagonal
    nominator = diff.dot(diff.T).dot(np.eye(n))
    denominator = add.T.dot(np.ones((n, 1)))
    return 0.5*nominator/denominator


class NLMNN():
    def __init__(self, l=1, mu=0.5, max_iter=1000, tol=1e-9):
        self.l = l
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x, y, A_init=None):
        n, d = x.shape

        # Initialize the A matrix if not specified
        if A_init is not None:
            self.A = A_init
        else:
            self.A = 10 * np.eye(d) + 0.01 * np.ones((d, d))

        distance = self.get_metric()

        for i in range(self.max_iter):

        distance(x[0, :].reshape(-1, d), x[1, :].reshape(-1, d))
        a = 1

    def get_metric(self):
        def D(x, y):
            L = softmax(self.A, axis=0)
            dist = chi2_distance(x.dot(L), y.dot(L))
            return dist
        return D


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    lnmnn = NLMNN()
    lnmnn.fit(X, y)
