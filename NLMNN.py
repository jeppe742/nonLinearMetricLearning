import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
#from numba import jit


def chi2_distance(x_i, x_j):
    return 0.5*sum(((x_i - x_j)**2)/(x_i+x_j))


# @jit(nopython=True)
# def dChi2_dA(i, j, p, q, L, t):
#     d,_ = L.shape
#     result= ((t[i, j, p] * (X[i, q] - X[j, q]) - t[i, j, p]**2*(X[i, q] + X[j, q])/2))

#     for l in range(d):
#         result -=(L[l, q] * (t[i, j, l] * (X[i, q] - X[j, q]) - t[i, j, l]**2*(X[i, q] + X[j, q])/2))

#     return L[p, q]*result

# @jit(nopython=True)
# def _get_grad(X, t, L, target_neighbours, imposters, mu):
#     '''
#     Calculate the gradient for a given iteration

#     Args:
#         X ([n,d] matrix): input data
#     Returns:
#         dL_dA ([d,d] matrix): gradient of loss function with respect to each entry in the A matrix
#     '''
#     n= 150
#     d=2
#     #d=1
#     t = t

#     grad = np.zeros((d, d))
#     # return grad
#     for p in range(d):
#         for q in range(d):
#             #print(f"{p},{q}")
#             for i in range(n):
#                 for imp, j in enumerate(target_neighbours[i]):
#                     # Pull step
#                     grad[p, q] += dChi2_dA(i, j, p, q, L, t)

#                     # Push step
#                     for k in imposters[i,imp]:
#                         grad[p, q] += mu * (dChi2_dA(i, j, p, q, L, t) - dChi2_dA(i, k, p, q, L, t))
#     return grad


class NLMNN():
    def __init__(self, l=0, mu=1, lr=100, max_iter=2000, tol=1e-9, k=3, max_lr_reductions=50):
        self.l = l
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.k = k
        self._grad_sizes = []
        self.losses = []
        self.max_lr_reductions = max_lr_reductions

    def get_target_neighbours(self, X, y):
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

        # Set all entries from different label to infinity
        pairwise_distance[y != y.T] = float("inf")

        # Sort entries and pick first k
        target_neighbours = np.argpartition(pairwise_distance, self.k)[:, 0:self.k]

        return target_neighbours

    def get_imposters(self, X, y):
        '''
        Get the imposters for each target neighbour to each datapoint

        Args:
            X ([n,d] matrix): Input data, where n is the number of data points and d is the dimension
            y ([n] array): True class for the data points

        Returns:
            Imposters ([n, k][n'] array of lists): index [i,j][k] has the kth imposter for target neighbour j of data point i
        '''

        n, d = X.shape
        # make sure the second dim is expanded
        y = y.reshape(-1, 1)
        # Calculate pairwise distance, using our metric
        pairwise_distance = pairwise_distances(X, metric=self.metric)

        np.fill_diagonal(pairwise_distance, float("inf"))

        imposters = []

        for i in range(n):
            imposters.append([])
            for neighbour_idx, j in enumerate(self.target_neighbours[i]):
                imposters[i].append([])
                for k in range(n):
                    if pairwise_distance[i, k] <= pairwise_distance[i, j] + self.l and y[i] != y[k]:
                        imposters[i][neighbour_idx].append(k)

        return np.array(imposters)

    def get_grad(self, X):
        '''
        Calculate the gradient for a given iteration

        Args:
            X ([n,d] matrix): input data
        Returns:
            dL_dA ([d,d] matrix): gradient of loss function with respect to each entry in the A matrix
        '''
        # return _get_grad(X, self.t, self.L, self.target_neighbours, self.imposters, self.mu)
        n, d = X.shape

        t = self.t

        def dChi2_dA(i, j, p, q):
            return self.L[p, q]*(
                (t[i, j, p] * (X[i, q] - X[j, q]) - t[i, j, p]**2*(X[i, q] + X[j, q])/2)
                - sum(self.L[:, q] * (t[i, j, :] * (X[i, q] - X[j, q]) - t[i, j, :]**2*(X[i, q] + X[j, q])/2))
            )

        grad = np.zeros((d, d))

        for p in range(d):
            for q in range(d):
                for i in range(n):
                    for j, imposters in zip(self.target_neighbours[i], self.imposters[i]):
                        # Pull step
                        grad[p, q] += dChi2_dA(i, j, p, q)

                        # Push step
                        for k in imposters:
                            grad[p, q] += self.mu * (dChi2_dA(i, j, p, q) - dChi2_dA(i, k, p, q))
        return grad

    def get_loss(self, X):
        n, d = X.shape
        loss = 0
        for i in range(n):
            for j, imposters in zip(self.target_neighbours[i], self.imposters[i]):
                loss += self.metric(X[i], X[j])

                for k in imposters:
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
        for i in tqdm(range(self.max_iter)):
            # Update the list of imposters
            self.imposters = self.get_imposters(X, y)

            self.t = self.get_t(X)

            # Get gradient
            grad = self.get_grad(X)
            # print("Gradient")
            print(grad)
            print(f"imposters={len([lll for l in self.imposters for ll in l for lll in ll])}")
            self._grad_sizes.append(np.sum(abs(grad)))
            print(np.sum(abs(grad)))
            A_tmp = self.A
            # Make sure the gradient step isn't too large, leading to divergence
            for i_step in range(self.max_lr_reductions):
                self.A = A_tmp - self.lr*grad
                self.L = self.calculate_L()
                loss = self.get_loss(X)
                if loss < best_loss:
                    best_loss = loss
                    self.losses.append(loss)

                    break

                else:
                    print(f"Gradient step too large, halfing learning rate   lr={self.lr:.2E}     loss = {loss}")
                    self.lr /= 2
                if i_step == (self.max_lr_reductions-1):
                    print("Could not find a learning rate that cause gradient step to improve loss...")
                    return

            #print("updated A")
            # print(self.A)
            print("updated L")
            print(self.L)
            print(f"updated loss = {loss}")

    def metric(self, x, y):
        dist = chi2_distance(x.dot(self.L), y.dot(self.L))
        return dist


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_wine
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    iris = load_iris()
    X = iris.data
    X = X[:, 0:2]
    X = X/np.sum(X, axis=1, keepdims=True)
    y = iris.target
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=2)

    # create reference KNN
    C1 = KNeighborsClassifier(n_neighbors=3)
    C1.fit(X_train, y_train)

    # train NLMNN transformation
    nlmnn = NLMNN()
    nlmnn.fit(X, y)

    C2 = KNeighborsClassifier(n_neighbors=3, metric=nlmnn.metric)
    C2.fit(X_train, y_train)
    print(f"normal KNN acc={C1.score(X_test, y_test)}")
    print(f"NLMNN  KNN acc={C2.score(X_test, y_test)}")

    plt.figure()
    plt.plot(nlmnn.losses)

    plt.figure()
    plt.plot(nlmnn._grad_sizes)
    plt.show()

    # h = .005
    # cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # # Plot the decision boundary. For that, we will assign a color to each
    # # point in the mesh [x_min, x_max]x[y_min, y_max].
    # x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    # y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))
    # Z = C1.predict(np.c_[xx.ravel(), yy.ravel()])

    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # # Plot also the training points
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold,
    #             edgecolor='k', s=20)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())

    # Z = C2.predict(np.c_[xx.ravel(), yy.ravel()])

    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # # Plot also the training points
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold,
    #             edgecolor='k', s=20)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())

    # plt.show()
