from NLMNN import NLMNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import scipy.io
import glob
from metric_learn import LMNN

classes = {
    'back_pack':0, 
    'bike':1,
    'calculator':2,
    'bike_helmet':3,
    'bookcase':4,
    'bottle':5,
    'calculator':6,
    'desk_chair':7,
    'desk_lamp':8,
    'desktop_computer':9
    }
targets = []
X = []
for file in glob.glob('data/webcam/*/*/*.mat'):
    
    target = file.split('/')[3]
    if target in classes:
        targets.append(classes[target])
        X.append(scipy.io.loadmat(file)['histogram'])
X = np.asarray(X, dtype=np.float).squeeze()
y = np.asarray(targets)

n,d = X.shape

X = X/np.sum(X, axis=1, keepdims=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# create reference KNN
C1 = KNeighborsClassifier(n_neighbors=3)
C1.fit(X_train, y_train)

C2 = KNeighborsClassifier(n_neighbors=3, metric='mahalanobis',
                        metric_params=dict(VI=np.eye(d)))
C2.fit(X_train, y_train)

lmnn = LMNN(k=3, max_iter=10000)
lmnn.fit(X_train, y_train)
C3 = KNeighborsClassifier(n_neighbors=3, metric='mahalanobis',
                        metric_params=dict(VI=lmnn.metric()))
C3.fit(X_train, y_train)
# train NLMNN transformation

nlmnn = NLMNN()
nlmnn.L = np.eye(d)
C4 = KNeighborsClassifier(n_neighbors=3, metric=nlmnn.metric)
C4.fit(X_train, y_train)


A= np.random.rand(d,d)
A = np.ones((d,d))+ np.random.rand(d,d)*0.01
#A = np.eye(d)

nlmnn2 = NLMNN(l=2e-9, jit=False) #distances are extremely small, so take a small margin
nlmnn2.fit(X_train, y_train, A_init=A)

C5 = KNeighborsClassifier(n_neighbors=3, metric=nlmnn2.metric)
C5.fit(X_train, y_train)



print(f"normal           KNN train acc={C1.score(X_train, y_train):.3f},  test acc={C1.score(X_test, y_test)}")
print(f"untrained LMNN   KNN train acc={C2.score(X_train, y_train):.3f},  test acc={C2.score(X_test, y_test)}")
print(f"LMNN             KNN train acc={C3.score(X_train, y_train):.3f},  test acc={C3.score(X_test, y_test)}")
print(f"untrained NLMNN  KNN train acc={C4.score(X_train, y_train):.3f},  test acc={C4.score(X_test, y_test)}")
print(f"trained   NLMNN  KNN train acc={C5.score(X_train, y_train):.3f},  test acc={C5.score(X_test, y_test)}")


nlmnn2.plot_debug()