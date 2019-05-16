import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from numba import jit, njit, prange

def chi2_distance(x_i, x_j):
    return 0.5*sum(((x_i - x_j)**2)/(x_i+x_j+1e-18))


@njit()
def _dChi2_dL(i, j, p, q, t, X):
    return t[i,j,q]*(X[i,p]- X[j,p]) - 0.5*t[i,j,q]**2*(X[i,p]+X[j,p])

@njit()
def dChi2_dA(i, j, p, q, L, t, X):
    #L is transposed compared to the paper. 
    d,r = L.shape
    result = ((t[i, j, q] * (X[i, p] - X[j, p]) - t[i, j, q]**2*(X[i, p] + X[j, p])/2))

    for l in range(r):
        result -= (L[p, l] * (t[i, j, l] * (X[i, p] - X[j, p]) - t[i, j, l]**2*(X[i, p] + X[j, p])/2))

    return L[p, q]*result


@njit(parallel=True)
def _metric(x, y, L):
    x = x.dot(L)
    y = y.dot(L)
    tmp = 0.0
    for i in range(len(x)):
        tmp += ((x[i] - y[i])**2)/(x[i] + y[i] + 1e-18)
    return 0.5*tmp

def simplex_projection(Y):
    n,d = Y.shape
    X = - np.sort(-Y, axis=1)
    X_tmp = (np.cumsum(X,axis=1)-1).dot(np.diag(1/np.arange(1,d+1)))  
    Z = np.maximum(Y-X_tmp[np.arange(0,n),np.sum(X>X_tmp,axis=1)-1].reshape(-1,1),0)
    return Z

@njit(parallel=True)
def _get_grad(X, t, L, target_neighbours, imposters, mu, r, use_softmax):
    '''
    Calculate the gradient for a given iteration

    Args:
        X ([n,d] matrix): input data
    Returns:
        dL_dA ([d,d] matrix): gradient of loss function with respect to each entry in the A matrix
    '''
    n, d = X.shape

    grad = np.zeros((d, r))

    for p in prange(d):
        for q in range(r):

            for i in range(n):
                for imp, j in enumerate(target_neighbours[i]):
                    # Pull step
                    if use_softmax:
                        tmp = dChi2_dA(i, j, p, q, L, t, X)
                    else:
                        tmp = _dChi2_dL(i, j, p, q, t, X)

                    grad[p, q] += tmp

                    # Push step
                    for k in imposters[i, imp]:
                        if k >= 0:
                            if use_softmax:
                                grad[p, q] += mu * (tmp - dChi2_dA(i, k, p, q, L, t, X))
                            else:
                                grad[p, q] += mu * (tmp - _dChi2_dL(i, k, p, q, t, X))

    return grad


@njit(parallel=True)
def _get_loss(X, target_neighbours, imposters, L, mu, l):
    '''
    Calculate the loss, given the current L matrix

    Args:
        X ([n,d] matrix): input data
    Returns:
        loss (float): Loss value
    '''
    n, d = X.shape

    #Allocate loss as a numpy array, which is needed for the jit to parallize the loop apparently
    loss = np.zeros((n))
    for i in prange(n):
        for imp, j in enumerate(target_neighbours[i]):

            tmp = _metric(X[i], X[j], L)
            loss[i] += tmp

            for k in imposters[i, imp]:
                if k >= 0:
                    loss[i] += mu * (l + tmp - _metric(X[i], X[k], L))
    return loss.sum()


class NLMNN():
    def __init__(self, 
                 l=0.01, 
                 mu=1, 
                 lr=1, 
                 max_iter=200, 
                 tol=0.01, 
                 k=3, 
                 auto_step_size=True, 
                 max_lr_reductions=20, 
                 jit=True, 
                 r=None, 
                 A_init=None,
                 use_softmax=True):
        self.l = l
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.k = k
        self._grad_sizes = []
        self.losses = []
        self.num_imposters = []
        self.auto_step_size = auto_step_size
        self.max_lr_reductions = max_lr_reductions
        self.jit = jit
        self.r = r
        self.t_get_imposter = []
        self.t_get_grad = []
        self.A_init = A_init
        self.use_softmax = use_softmax

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
        pairwise_distance = pairwise_distances(X, metric=lambda X,y: _metric(X,y,self.L), n_jobs=1)

        np.fill_diagonal(pairwise_distance, float("inf"))

        imposters = []
        #Numba doesn't support nested lists, so we have to calculate this on the python side
        for i in range(n):
            imposters.append([])
            for neighbour_idx, j in enumerate(self.target_neighbours[i]):
                imposters[i].append([])
                for k in range(n):

                    if pairwise_distance[i, k] <= pairwise_distance[i, j] + self.l and y[i] != y[k]:
                        imposters[i][neighbour_idx].append(k)

        # Convert to padded numpy array. This might use unnessecary memory for large datasets
        max_num_imposters = len(max([max(ll, key=len) for ll in imposters], key=len))
        imposters_np = np.ones((n, self.k, max_num_imposters), dtype=np.int)*-1

        for i in range(n):
            for j in range(self.k):
                imposters_np[i, j, 0:len(imposters[i][j])] = imposters[i][j]

        return imposters_np

    def get_grad_approx(self, X, y, delta=0.001):
        '''
        Calculate the gradient for a given iteration

        Args:
            X ([n,d] matrix): input data
        Returns:
            dL_dA ([d,d] matrix): gradient of loss function with respect to each entry in the A matrix
        '''

        n, d = X.shape

        grad = np.zeros((d, d))

        A_tmp = self.A.copy()
        f_tmp = self.get_loss(X)
        L_tmp = self.L.copy()
        imposters_tmp = self.imposters.copy()
        # Calculate the gradient for each index in A
        for p in range(d):
            for q in range(d):

                # add a small permutation to A, and update all dependent variables
                self.A[p, q] += delta
                self.L = self.calculate_L()
                self.imposters = self.get_imposters(X, y)
                # calculate permuted loss
                f_delta = self.get_loss(X)
                # approximate gradient by finite difference
                grad[p, q] = (f_delta-f_tmp)/delta

                # reset variables
                self.A = A_tmp
                self.L = L_tmp
                self.imposters = imposters_tmp
        return grad

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

        if self.jit:
            
            grad = _get_grad(X, t, self.L, self.target_neighbours, self.imposters, self.mu, self.r, self.use_softmax)
            print(grad)
            print(_get_grad(X, t, self.L, self.target_neighbours, self.imposters, self.mu, self.r, False))
            return grad

        # Define partial derivative function for chainrule calculation

        def dChi2_dA(i, j, p, q):
            return self.L[p, q]*(
                (t[i, j, p] * (X[i, q] - X[j, q]) - t[i, j, p]**2*(X[i, q] + X[j, q])/2)
                - sum(self.L[:, q] * (t[i, j, :] * (X[i, q] - X[j, q]) - t[i, j, :]**2*(X[i, q] + X[j, q])/2))
            )

        grad = np.zeros((d, self.r))

        # Calculate the gradient for each index in A
        for p in range(d):
            for q in range(self.r):
                # Loop over all datapoints
                for i in range(n):
                    # Loop over all target neighbours, and their corresponding imposters
                    for j, imposters in zip(self.target_neighbours[i], self.imposters[i]):
                        # Pull step
                        grad[p, q] += dChi2_dA(i, j, p, q)

                        # Push step
                        for k in imposters:
                            if k >= 0:
                                grad[p, q] += self.mu * (dChi2_dA(i, j, p, q) - dChi2_dA(i, k, p, q))
        return grad

    def get_loss(self, X):
        '''
        Calculate the loss, given the current L matrix

        Args:
            X ([n,d] matrix): input data
        Returns:
            loss (float): Loss value
        '''

        if self.jit:
            loss = _get_loss(X, self.target_neighbours, self.imposters, self.L, self.mu, self.l)
            
            return loss

        n, d = X.shape
        loss = 0
        for i in range(n):
            for j, imposters in zip(self.target_neighbours[i], self.imposters[i]):
                loss += self.metric(X[i], X[j])

                for k in imposters:
                    if k >= 0:
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
        x_bar = X.dot(self.L).reshape(n, 1, self.r)

        # by adding one extra dimension, we should get the elementwise difference between each datapoint
        x_bar_diff = x_bar - x_bar.reshape(1, n, self.r)
        x_bar_add = x_bar + x_bar.reshape(1, n, self.r)

        t = x_bar_diff/x_bar_add
        return t

    def calculate_L(self):
        return softmax(self.A, axis=1)

    def fit(self, X, y, verbose=False, use_tqdm=False):
        n, d = X.shape
        if self.r is None:
            self.r = d
        elif self.r > d:
            raise AssertionError(f'Projected dimension is higher than original  r={self.r}, d={d}')

        # Initialize the A matrix if not specified
        if self.A_init is not None:
            self.A = self.A_init
        else:
            self.A = (10 * np.eye(d) + 0.01 * np.ones((d, d)))[:,:self.r] #Only take r columns 

        # Update L from A
        self.L = self.calculate_L()
        self.t = self.get_t(X)
        # Find the target neighbours

        self.target_neighbours = self.get_target_neighbours(X, y)

        self.imposters = self.get_imposters(X, y)
        total_imposters = np.sum(self.imposters>=0)
        #print(f"imposters={total_imposters}")
        best_loss = float("inf")
        for i in tqdm(range(self.max_iter),disable=(not use_tqdm)):

            # Get gradient
            grad = self.get_grad(X)

            total_imposters = np.sum(self.imposters>=0)
            self.num_imposters.append(total_imposters)
            grad_size = np.sum(abs(grad))
            self._grad_sizes.append(grad_size)
            if verbose:
                    
                print("\nGradient")
                print(grad)
                print(f"Gradient size = {grad_size}")
                print(f"imposters={total_imposters}")
            
            
            if self.auto_step_size:
                # Create copy of A before gradient update
                A_tmp = self.A.copy()
                L_tmp = self.L.copy()
                # Make sure the gradient step isn't too large, leading to divergence
                for i_step in range(self.max_lr_reductions):
                    if self.use_softmax:
                        self.A = A_tmp - self.lr*grad
                        self.L = self.calculate_L()
                    else:
                        self.L = L_tmp - self.lr*grad
                        self.L = simplex_projection(self.L)
                    # Update the list of imposters
                    self.imposters = self.get_imposters(X, y)
                    loss = self.get_loss(X)

                    if loss <= best_loss:
                        if (best_loss-loss) < self.tol:
                            if verbose:
                                print("tolerance reached")
                            return
                        best_loss = loss
                        self.losses.append(loss)
                        self.lr *= 1.01

                        break

                    else:
                        if verbose:
                            print(f"Gradient step too large, halfing learning rate   lr={self.lr:.2E}     loss = {loss}")
                        self.lr /= 2
                    if i_step == (self.max_lr_reductions-1):
                        if verbose:
                            print("Could not find a learning rate that cause gradient step to improve loss...")
                        return
            else:
                #Update A and all dependent variables
                self.A = self.A - self.lr*grad
                self.L = self.calculate_L()
                self.t = self.get_t(X)
                self.imposters = self.get_imposters(X,y)
                loss = self.get_loss(X)
                self.losses.append(loss)

            if verbose:
                print("updated L")
                print(self.L)
                print(f"updated loss = {loss}")

    def metric(self, x, y):
        dist = chi2_distance(x.dot(self.L), y.dot(self.L))
        return dist

    def plot_debug(self):
        import matplotlib.pyplot as plt
        print(f"loss = {self.losses[-1]}")
        print(f"number of imposters = {self.num_imposters[-1]}")
        plt.figure()
        plt.subplot(221)
        plt.plot(self.losses)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')

        plt.subplot(222)
        plt.plot(self._grad_sizes)
        plt.ylabel('L1 norm of gradient')
        plt.xlabel('Iterations')

        plt.subplot(223)
        plt.plot(self.num_imposters)
        plt.ylabel('Number of imposters')
        plt.xlabel('Iterations')

        plt.subplot(224)
        #plot transpose L, since the implementation uses xL instead of Lx as in the paper
        plt.imshow(self.L.T, aspect='auto')
        plt.colorbar()
        plt.show()
    

if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from metric_learn import LMNN
    import glob
    import scipy

    # x1 = np.array([0.5, 0.5]).reshape(2,1) + np.random.randn(2,10)*0.01
    # x2 = np.array([0.5, 0.5]).reshape(2,1) + np.random.randn(2,10)*0.01

    # X = np.hstack((x1, x2)).transpose(1,0)
    # #X = np.array([[0,0],[]])
    #iris = load_iris()
    # iris = load_wine()
    # X = iris.data
    # X = X#[:, 0:7]
    # n, d = X.shape
    # np.random.seed(1)
    # #sX = X + np.random.randn(n,d)

    # # X = np.array([[0,0,1],[0,0.5,0.5],[0,1,0],
    # #              [1/3,0,2/3],[1/3,1/3,1/3],[1/3,2/3,0],
    # #              [2/5,0,3/5],[2/5,1/5,2/5],[2/5,3/5,0]])
    # #X = np.meshgrid(np.linspace(1,3,3),np.linspace(1,3,3))

    # X = X/np.sum(X, axis=1, keepdims=True)
    # #y = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
    # # y = np.array([0,0,0,1,1,1,2,2,2])
    # y = iris.target  # [:50]

    classes = {
        'back_pack':1,
        'bike':2,
        'calculator':3,
        'headphones':4,
        'keyboard':5,
        'laptop_computer':6,
        'monitor':7,
        'mouse':8,
        'mug':9,
        'projector':10,
    }
    X = []
    targets=[]
    for file in glob.glob('data/webcam/*/*/*.mat'):
        
        target = file.split('/')[3]
        if target in classes:
            targets.append(classes[target])
            X.append(scipy.io.loadmat(file)['histogram'])
    X = np.asarray(X, dtype=np.float).squeeze()
    y = np.asarray(targets)

    n,d = X.shape

    X = X/np.sum(X, axis=1, keepdims=True)

    clf3 = NLMNN(k=3, lr=1, use_softmax=True, max_iter=2, r=10)
    clf3.fit(X,y, verbose=False)
    # plt.show()

    #pca = PCA()
    # pca.fit(X)
    # #X = pca.transform(X)
    # # split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # # create reference KNN
    # C1 = KNeighborsClassifier(n_neighbors=3)
    # C1.fit(X_train, y_train)

    # C2 = KNeighborsClassifier(n_neighbors=3, metric='mahalanobis',
    #                       metric_params=dict(VI=np.eye(d)))
    # C2.fit(X_train, y_train)

    # lmnn = LMNN(k=3, max_iter=10000)
    # lmnn.fit(X_train, y_train)
    # C3 = KNeighborsClassifier(n_neighbors=3, metric='mahalanobis',
    #                       metric_params=dict(VI=lmnn.metric()))
    # C3.fit(X_train, y_train)
    # # train NLMNN transformation

    # nlmnn = NLMNN()
    # nlmnn.L = np.eye(d)
    # C4 = KNeighborsClassifier(n_neighbors=3, metric=nlmnn.metric)
    # C4.fit(X_train, y_train)
    

    # A= np.random.rand(d,d)
    # A = np.ones((d,d))+ np.random.rand(d,d)*0.01
    # #A = np.eye(d)
    # nlmnn2 = NLMNN()
    # nlmnn2.fit(X_train, y_train)

    # C5 = KNeighborsClassifier(n_neighbors=3, metric=nlmnn2.metric)
    # C5.fit(X_train, y_train)

    

    # print(f"normal           KNN train acc={C1.score(X_train, y_train):.3f},  test acc={C1.score(X_test, y_test)}")
    # print(f"untrained LMNN   KNN train acc={C2.score(X_train, y_train):.3f},  test acc={C2.score(X_test, y_test)}")
    # print(f"LMNN             KNN train acc={C3.score(X_train, y_train):.3f},  test acc={C3.score(X_test, y_test)}")
    # print(f"untrained NLMNN  KNN train acc={C4.score(X_train, y_train):.3f},  test acc={C4.score(X_test, y_test)}")
    # print(f"trained   NLMNN  KNN train acc={C5.score(X_train, y_train):.3f},  test acc={C5.score(X_test, y_test)}")
    
   
    # nlmnn2.plot_debug()

    #X2 = X.dot(nlmnn.L)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X2[:,0],X2[:,1],X2[:,2],c=y,marker='x', label='X*L')
    # ax.scatter(X[:,0],X[:,1],X[:,2],c=y,marker='o', label='X')
    # plt.xlim(0,0.5)
    # plt.ylim(0,1)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()

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
