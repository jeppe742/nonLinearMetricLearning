
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:01:12 2019
 
@author: nsde
"""
 
#%%
import glob
import cv2, os, re
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
 
class args:
    n_features = 10
    n_files = 100 # set to 7202 for all files
    use_online = True



def get_features(f):
    # Read image and extract label from filename
    img = cv2.imread(f)
    label = f.split('/')[2]
        
    # We use AKAZE since SIFT has moved to a external library,
    # that is a pain to install
    # We could also use something like HOG descriptor
    alg = cv2.KAZE_create()
    alg.setThreshold(1e-3)
        
    # Detech keypoints
    kps = alg.detect(img)
    if len(kps)>0:
        # Extract descriptors at each keypoint
        _, dsc = alg.compute(img, kps)
        return label, dsc


if __name__ == '__main__':
    img_dsc = [ ]
    labels = [ ]
    #d = 'coil-100' # download from http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php
    files = glob.glob('data/asl_alphabet_train/*/*.jpg')
    print('Extraction descriptors')
    results=Parallel(n_jobs=30, prefer='threads')(delayed(get_features)(f) for f in tqdm(files))
    
    labels, img_dsc = zip(*[result for result in results if result is not None])

         
    # Number of images
    N = len(labels)
     
    # Do train/test split (10% for test)
    idx = np.random.permutation(N)
    idx_train = idx[:int(N*9/10)]
    idx_test = idx[int(N*9/10):]
     
    dsc_train = [img_dsc[i] for i in idx_train]
    dsc_test = [img_dsc[i] for i in idx_test]
    labels_train = [labels[i] for i in idx_train]
    labels_test = [labels[i] for i in idx_test]
     
    # Kmeans on training set to find cluster centers
    # NOTE: You may want to set the use_online argument to true, as the
    # standard kmeans algorithm may take a very long time
    print('Fitting Kmeans')
    cluster_alg = MiniBatchKMeans(n_clusters = args.n_features, verbose=True) if args.use_online \
                    else KMeans(n_clusters = args.n_features) 
    cluster_alg.fit(np.concatenate(dsc_train))
    cluster_alg.verbose = False
    # Build histograms, by for each image calculate how many descriptors
    # that are close to each cluster center
    print('Building histograms')
    X_train = np.zeros((len(idx_train), args.n_features))
    X_test = np.zeros((len(idx_test), args.n_features))
     
    for i, dsc in enumerate(dsc_train):
        cluster_idx = cluster_alg.predict(dsc_train[i])
        X_train[i] = np.bincount(cluster_idx, minlength=args.n_features)
     
    for i, dsc in enumerate(dsc_test):
        cluster_idx = cluster_alg.predict(dsc_test[i])
        X_test[i] = np.bincount(cluster_idx, minlength=args.n_features)
         
    # Normalize so we actually have histograms
    X_train = X_train / X_train.sum(axis=1, keepdims=True)
    X_test = X_test / X_test.sum(axis=1, keepdims=True)
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)
     
    # Save res, can be loaded using np.load
    np.savez('my_histograms_' + str(args.n_features), 
             X_train = X_train,
             X_test = X_test,
             y_train = y_train,
             y_test = y_test)
     
    # Plot histogram from different classes
    # fig, ax = plt.subplots(5, 2)
    # for i, rand_idx in enumerate(np.random.choice(int(N*9/10), size=5)):
    #     ax[i, 0].imshow(cv2.imread(d + '/' + files[idx[rand_idx]]))
    #     ax[i, 0].axis('off')
    #     ax[i, 1].bar(np.arange(args.n_features), X_train[rand_idx])
    # plt.show()
