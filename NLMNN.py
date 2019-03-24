import numpy as np
from scipy.special import softmax
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances


def chi2_distance(x_i,x_j):
    _,d = x_i.shape

    return 0.5*sum((x_i- x_j)**2)/sum(x_i+x_j)

class NLMNN():
    def __init__(self):
        pass
    def get_target_neighbours(self,X,y,k=5):

        target_neighbours = np.zeros((self.num_classes,k,2))


        pairwise_distance = pairwise_distances(X,X)
        pairwise_distance[np.tril_indices(n)] = float("inf")
        for label in range(num_classes):
            pd_tmp = pairwise_distance.copy()
            pd_tmp[y!=label, :] = float("inf")
            #pd_tmp[y!=label, y==label] = float("inf")
            for i in range(k):
                idx = np.unravel_index(np.argmin(pd_tmp),pd_tmp.shape)
                target_neighbours[label,i,:] = idx
                pd_tmp[idx] = float("inf")


iris = load_iris()
X = iris.data
y = iris.target

n,d = X.shape

num_classes = len(np.unique(y))
k = 5


A = np.eye(d)
L = softmax(A,axis=0)

x_bar = X.dot(L).reshape(n,d)

t = np.zeros((n,n,d))

for i in range(n):
    for j in range(n):
        for p in range(d):
            t[i,j,p] = (x_bar[i,p] - x_bar[j,p])/(x_bar[i,p] + x_bar[j,p])


# for p in range(d):
#     for q in range(d):



#def grad(X,y,target_neighbours):
