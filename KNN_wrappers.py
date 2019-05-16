from sklearn.neighbors import KNeighborsClassifier
from metric_learn import LMNN
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from NLMNN import NLMNN


class LMNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k=3, pca=None, train=True, mu=0.5):
        self.k = k
        self.train = train
        self.pca = pca
        self.pca_trasform = None
        self.mu = mu
        self.lmnn = LMNN(k=k,use_pca=False, max_iter=10000, regularization=mu)
    def fit(self, x, y=None):
        n,d = x.shape
        

        if self.pca is not None:
            pca = PCA(n_components=self.pca)
            pca.fit(x)
            self.pca_trasform = pca.transform
            x = pca.transform(x)
        if self.train:
            self.lmnn.fit(x,y)
            self.knn = KNeighborsClassifier(n_neighbors=self.k, metric='mahalanobis',
                          metric_params=dict(VI=self.lmnn.metric()), n_jobs=-1)
        else:
            self.knn = KNeighborsClassifier(n_neighbors=self.k)

        
        self.knn.fit(x,y)

        return self
    def predict(self, x, y=None):
        if self.pca_trasform is not None:
            x = self.pca_trasform(x)
        return self.knn.predict(x,y)
    
    def score(self, x,y=None):
        if self.pca_trasform is not None:
            x = self.pca_trasform(x)
        return self.knn.score(x,y)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            if parameter == 'mu':
                setattr(self.lmnn, 'regularization', value)
        return self



class Chi2Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k=3, r=None):
        self.k = k
        self.r = r
    def fit(self, x, y=None):
        n,d = x.shape
        self.nlmnn = NLMNN(k=self.k, r=self.r)
        
        if self.r is not None:
            self.nlmnn.L = np.eye(d,self.r)
        else:
            self.nlmnn.L = np.eye(d)
        self.knn = KNeighborsClassifier(n_neighbors=self.k, metric=self.nlmnn.metric, n_jobs=-1)
        self.knn.fit(x,y)
        
        return self
    def predict(self, x, y=None):
        return self.knn.predict(x, y)
    
    def score(self, x, y=None):
        return self.knn.score(x,y)


class NLMNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k=3, r=None, A_init=None, mu=1, l=0.01, lr=1, use_softmax=True, max_lr_reductions=20):
        self.k=k
        self.A_init = A_init
        self.r = r
        self.mu = mu
        self.l = l
        self.A_init = A_init
        self.lr = lr
        self.use_softmax = use_softmax
        self.max_lr_reductions = max_lr_reductions
        self.nlmnn = NLMNN(k=k, r=r, mu=mu, l=l, A_init=A_init, lr=lr, use_softmax=use_softmax, max_lr_reductions=max_lr_reductions)
    def fit(self, x, y=None, verbose=False):
        
        self.nlmnn.fit(x,y, verbose=verbose)
        self.knn = KNeighborsClassifier(n_neighbors=self.k, metric=self.nlmnn.metric, n_jobs=-1)
        self.knn.fit(x,y)
        
        return self
    def predict(self, x, y=None):
        return self.knn.predict(x, y)
    
    def score(self, x, y=None):
        return self.knn.score(x,y)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            setattr(self.nlmnn, parameter, value)
        return self