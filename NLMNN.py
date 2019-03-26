import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform

def chi2_distance(x_i, x_j):
    d = x_i.shape

    return 0.5*sum((x_i - x_j)**2)/sum(x_i+x_j)


# def chi2_distance(x, y):
#     """
#     Calculates the chi2 distance between two multivariate inputs

#     Args:
#         x: [n', d] matrix, where the number of datapoints n' either has to be n or 1, and d is the dimension 
#         y: [n, d] matrix, where n is the number of datapoints 
#     """
#     n, d = y.shape

#     diff = x-y
#     add = x+y
#     print(X)
#     print(diff)
#     # The first dot calculates (x-y)^2, but this will give a [n,n] matrix with all cross terms.
#     # We are however only interested in the diagonal
#     nominator = diff.dot(diff.T).dot(np.eye(n))
#     print(nominator)
#     denominator = add.dot(np.ones((d, 1)))
#     print(nominator)
#     return 0.5*nominator/denominator

class NLMNN():
    def __init__(self):
        pass


def get_target_neighbours(X, y, k=5):
    '''
    Get the index of the k closest neighbours to each point in the dataset, for each label

    Args:
        X ([n, d] matrix): Input data, where n is the number of data points and d is the dimension
        y ([n] array): The class label for each datapoint
        k (int): number of neighbours for each datapoint

    Returns:
        target_neighbours ([n, k, num_classes]): index [i,j,k] has the index of the jth neighbour of datapoint i, with label k 

    '''
    n, d = X.shape
    num_classes = len(np.unique(y))

    target_neighbours = np.zeros((n, k, num_classes))

    # calculate pairwise distance
    pairwise_distance = pairwise_distances(X, X)

    # Fill diagonal with infinity, since we want to ignore these
    np.fill_diagonal(pairwise_distance, float("inf"))

    for label in range(num_classes):
        pd_tmp = pairwise_distance.copy()
        # Set all entries from different label to infinity
        pd_tmp[y != label, :] = float("inf")

        # Sort entries and pick first k
        idx = np.argpartition(pd_tmp, k)[:, 0:k]

        target_neighbours[:, :, label] = idx
    return target_neighbours


def get_imposters(X, y):
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
    pairwise_distance = pairwise_distances(X, metric = chi2_distance)
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


def metric(x, y, A):
    L = softmax(A, axis=0)
    dist = chi2_distance(x.dot(L), y.dot(L))
    return dist

iris = load_iris()
X = iris.data
y = iris.target

#get_target_neighbours(X, y)
#print(squareform(pdist(X,chi2_distance)))
asd=pairwise_distances(X, metric=chi2_distance)
imposters = get_imposters(X,y)
# n, d = X.shape

# num_classes = len(np.unique(y))
# k = 5


# A = np.eye(d)
# L = softmax(A, axis=0)

# x_bar = X.dot(L).reshape(n, d)

# t = np.zeros((n, n, d))

# for i in range(n):
#     for j in range(n):
#         for p in range(d):
#             t[i, j, p] = (x_bar[i, p] - x_bar[j, p])/(x_bar[i, p] + x_bar[j, p])


# for p in range(d):
#     for q in range(d):


# def grad(X,y,target_neighbours):
