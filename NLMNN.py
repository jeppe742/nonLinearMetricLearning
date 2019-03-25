import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances


def chi2_distance(x_i, x_j):
    _, d = x_i.shape

    return 0.5*sum((x_i - x_j)**2)/sum(x_i+x_j)


class NLMNN():
    def __init__(self):
        pass


def get_target_neighbours(X, y, k=5):
    '''
    Get the k closest neighbours to each point in the dataset, for each label

    Arg:
        X ([n, d] matrix): Input data, where n in the number of points and d is the dimension
        y ([n] array): The class label for each datapoint
        k (int): number of neighbours for each datapoint

    Returns:
        target_neighbours ([n, k, num_classes]): index [i,j,k] has the jth neighbour of datapoint i, with label k 

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

        # Sort entries and pick first 5
        idx = np.argpartition(pd_tmp, k)[:, 0:5]

        target_neighbours[label, :, :] = idx
    return target_neighbours


def get_imposters(x, y):
    '''

    '''

    pass


iris = load_iris()
X = iris.data
y = iris.target

get_target_neighbours(X, y)

n, d = X.shape

num_classes = len(np.unique(y))
k = 5


A = np.eye(d)
L = softmax(A, axis=0)

x_bar = X.dot(L).reshape(n, d)

t = np.zeros((n, n, d))

for i in range(n):
    for j in range(n):
        for p in range(d):
            t[i, j, p] = (x_bar[i, p] - x_bar[j, p])/(x_bar[i, p] + x_bar[j, p])


# for p in range(d):
#     for q in range(d):


# def grad(X,y,target_neighbours):
