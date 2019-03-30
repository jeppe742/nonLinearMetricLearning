import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise_distances


def chi2_distance(x_i, x_j):
    d = x_i.shape

    return 0.5*sum((x_i - x_j)**2)/sum(x_i+x_j)


class NLMNN():
    def __init__(self, l=0.1, mu=0, lr=10, max_iter=1000, tol=1e-9):
        self.l = l
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr

    def get_target_neighbours(self, X, y, k=15):
        '''
        Get the k closest neighbours to each point in the dataset, for each label

        Arg:
            X ([n, d] matrix): Input data, where n in the number of points and d is the dimension
            y ([n] array): The class label for each datapoint
            k (int): number of neighbours for each datapoint

        Returns:
            target_neighbours ([n, k]): index [i,j] has the jth closest neighbour of point i, with same class

        '''

        y = y.reshape(-1, 1)

        n, d = X.shape

        # Calculate pairwise distance
        pairwise_distance = pairwise_distances(X, X)

        # Fill diagonal with infinity, since we want to ignore these
        np.fill_diagonal(pairwise_distance, float("inf"))

        # for label in range(num_classes):
        pd_tmp = pairwise_distance.copy()
        # Set all entries from different label to infinity
        pd_tmp[y != y.T] = float("inf")

        # Sort entries and pick first k
        target_neighbours = np.argpartition(pd_tmp, k)[:, 0:k]

        return target_neighbours

    def get_imposters(self, X, y):
        '''
        Get the imposters for each datapoint, defined as points classified wrongly as the same class as the datapoint

        Args:
            X ([n,d] matrix): Input data, where n is the number of data points and d is the dimension
            y ([n] array): True class for the data points

        Returns:
            Imposters ([n, n'] list of lists): index [i,j] has the jth imposter for data point i
        '''

        n, d = X.shape
        # make sure the second dim is expanded
        y = y.reshape(-1, 1)
        # Calculate pairwise distance, using our metric
        pairwise_distance = pairwise_distances(X, metric=self.metric)
        np.fill_diagonal(pairwise_distance, float("inf"))

        # imposters should be misclassified by a margin l
        pairwise_distance[y != y.T] += self.l

        # Find the closest neighbour for each datapoint
        closest_neighbour = np.argmin(pairwise_distance, axis=1)

        # Find the indecies where the closest neighbour of a datapoint has a different class
        imposters_idx = np.flatnonzero(y[closest_neighbour] != y)

        imposters = [[] for _ in range(n)]

        for i, imposter in enumerate(imposters_idx):

            # The target neighbourhood of x_i are the points that have x_i as their closest neighbour
            imposters[imposter].append(i)

        return imposters

    def get_grad(self, X):
        '''
        Calculate the gradient for a given iteration

        Args:
            X ([n,d] matrix): input data
        Returns:
            dL_dA ([d,d] matrix): gradient of loss function with respect to each entry in the A matrix
        '''
        n, d = X.shape

        t = self.t

        def dChi2_dA(i, j, p, q):
            return self.L[p, q]*((t[i, j, p] * (X[i, q] - X[j, q]) - t[i, j, p]**2*(X[i, q] + X[j, q])/2)
                                 - sum(self.L[:, q] * (t[i, j, :] * (X[i, q] - X[j, q]) - t[i, j, :]**2*(X[i, q] + X[j, q])/2)))

        grad = np.zeros((d, d))
        # return grad
        for p in range(d):
            for q in range(d):

                for i in range(n):
                    for j in self.target_neighbours[i]:
                        # Pull step
                        grad[p, q] += dChi2_dA(i, j, p, q)

                        # Push step
                        for k in self.imposters[i]:
                            grad[p, q] += self.mu * (dChi2_dA(i, j, p, q) - dChi2_dA(i, k, p, q))
        return grad

    def get_loss(self, X):
        n, d = X.shape
        loss = 0
        for i in range(n):
            for j in self.target_neighbours[i]:
                loss += self.metric(X[i], X[j])

                for k in self.imposters[i]:
                    loss += self.mu * (self.l + self.metric(X[i], X[j]) - self.metric(X[i], X[k]))
        return loss

    def get_t(self, X):
        '''
        Calculate t as defined in Yang et al.

        Args:
            X ([n,d] matrix): input data
        Returns:
            t ([n,n,d] matrix): t matrix
        '''
        n, d = X.shape

        # Project X using L
        x_bar = X.dot(self.L).reshape(n, 1, d)

        # by adding one extra dimension, we should get the elementwise difference between each datapoint
        x_bar_diff = x_bar - x_bar.reshape(1, n, d)
        x_bar_add = x_bar + x_bar.reshape(1, n, d)

        t = x_bar_diff/x_bar_add
        return t

    def calculate_L(self):
        return softmax(self.A, axis=0)

    def fit(self, X, y, A_init=None):
        n, d = X.shape

        # Initialize the A matrix if not specified
        if A_init is not None:
            self.A = A_init
        else:
            self.A = 10 * np.eye(d) + 0.01 * np.ones((d, d))
        # Update L from A
        self.L = self.calculate_L()
        # Find the target neighbours
        self.target_neighbours = self.get_target_neighbours(X, y)

        best_loss = float("inf")
        for i in range(self.max_iter):
            # Update the list of imposters
            self.imposters = self.get_imposters(X, y)
            self.t = self.get_t(X)

            # Get gradient
            grad = self.get_grad(X)
            print("\nGradient")
            print(grad)
            print(np.sum(abs(grad)))
            A_tmp = self.A
            # Make sure the gradient step isn't too large, leading to divergence
            for ø in range(100):
                self.A = A_tmp - self.lr*grad
                self.L = self.calculate_L()
                loss = self.get_loss(X)
                if loss < best_loss:
                    best_loss = loss

                    break

                else:
                    print(f"Gradient step too large, halfing learning rate   lr={self.lr:.3f}     loss = {loss}")
                    self.lr /= 2
                # if ø == 99:
                #     return
            # Update L from new A
            #self.L = self.calculate_L()
            print("updated A")
            print(self.A)
            print("updated L")
            print(self.L)
            print(f"updated loss = {loss}")

    def metric(self, x, y):
        dist = chi2_distance(x.dot(self.L), y.dot(self.L))
        return dist


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    X = X/np.sum(X, axis=1, keepdims=True)
    y = iris.target

    nlmnn = NLMNN()
    # nlmnn.get_target_neighbours(X, y)
    nlmnn.fit(X, y)
