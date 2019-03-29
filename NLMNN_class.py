import numpy as np
from scipy.special import softmax
from sklearn.metrics import pairwise_distances

def chi2_distance(x_i, x_j):
    d = x_i.shape

    return 0.5*sum((x_i - x_j)**2)/sum(x_i+x_j)



class NLMNN():
    def __init__(self, l=1, mu=0.5, lr=0.1, max_iter=1000, tol=1e-9):
        self.l = l
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr

    def get_target_neighbours(self, X, y, k=5):
        '''
        Get the k closest neighbours to each point in the dataset, for each label

        Arg:
            X ([n, d] matrix): Input data, where n in the number of points and d is the dimension
            y ([n] array): The class label for each datapoint
            k (int): number of neighbours for each datapoint

        Returns:
            target_neighbours ([n, k, num_classes]): index [i,j,k] has the jth neighbour of datapoint i, with label k 

        '''

        y = y.reshape(-1,1)

        n, d = X.shape

        # Calculate pairwise distance
        pairwise_distance = pairwise_distances(X, X)

        # Fill diagonal with infinity, since we want to ignore these
        np.fill_diagonal(pairwise_distance, float("inf"))

        #for label in range(num_classes):
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
        #make sure the second dim is expanded
        y = y.reshape(-1,1)
        #Calculate pairwise distance, using our metric
        pairwise_distance = pairwise_distances(X, metric = self.metric)
        np.fill_diagonal(pairwise_distance, float("inf"))

        #Find the closest neighbour for each datapoint
        closest_neighbour = np.argmin(pairwise_distance, axis=1)

        #Find the indecies where the closest neighbour of a datapoint has a different class
        imposters_idx = np.flatnonzero(y[closest_neighbour]!=y)

        imposters = [[] for _ in range(n)]
        
        for i, imposter in enumerate(imposters_idx):
            
            #The target neighbourhood of x_i are the points that have x_i as their closest neighbour
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
        L = self.get_L()
        t = self.get_t(X)
        imposters = self.get_imposters(X,y)

        def dChi2_dA(i,j,p,q):
            return L[p,q]*( (t[i,j,p] * ( X[i,q] - X[j,q] ) - t[i,j,p]**2*(X[i,q] + X[j,q])/2 ) \
                    - sum(L[:,q] * (t[i,j,:] * (X[i,q] - X[j,q]) - t[i,j,:]**2*(X[i,q] + X[j,q])/2 )) )
        #TODO calculate loss function value?
        grad = np.zeros((d,d))
        #return grad
        for p in range(d):
            for q in range(d):
                
                for i in range(n):
                    for j in self.target_neighbours[i]:
                    #Pull step
                        grad[p,q] += dChi2_dA(i,j,p,q)
                    
                        #Push step
                        for k in imposters[i]:
                            grad[p,q] += self.mu * ( dChi2_dA(i,j,p,q) - dChi2_dA(i,k,p,q))
        return grad
    def get_t(self, X):
        '''
        Calculate t as defined in Yang et al.

        Args:
            X ([n,d] matrix): input data
        Returns:
            t ([n,n,d] matrix): t matrix
        '''
        n, d = X.shape
        
        #calculate L from A
        L = self.get_L()
        
        #Project X using L
        x_bar = X.dot(L).reshape(n,1,d)
        
        #by adding one extra dimension, we should get the elementwise difference between each datapoint
        x_bar_diff = x_bar - x_bar.reshape(1,n,d)
        x_bar_add = x_bar + x_bar.reshape(1,n,d)

        t = x_bar_diff/x_bar_add
        return t
    
    def get_L(self):
        return softmax(self.A,axis=0)

    def fit(self, x, y, A_init=None):
        n, d = x.shape

        # Initialize the A matrix if not specified
        if A_init is not None:
            self.A = A_init
        else:
            self.A = 10 * np.eye(d) + 0.01 * np.ones((d, d))

        distance = self.metric

        self.target_neighbours = self.get_target_neighbours(x, y)

        for i in range(self.max_iter):
            grad = self.get_grad(X)
            self.A -= self.lr*grad

    def metric(self, x, y):
        L = self.get_L()
        dist = chi2_distance(x.dot(L), y.dot(L))
        return dist



if __name__ == '__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    nlmnn = NLMNN()
    nlmnn.get_target_neighbours(X,y)
    nlmnn.fit(X, y)
