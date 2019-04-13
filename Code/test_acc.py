import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from skimage import io, transform
from joblib import Parallel, delayed

em = np.load('train_embeddings.npy')[10:,:]
em2 = np.load('test_embeddings.npy')[10:,:]

lab1 = np.load('train_lab.npy')
lab2 = np.array(pd.Series(np.load('test_lab.npy')).map(Tiny.class_dict)).ravel()

print('fitting NN')
neigh = NearestNeighbors(n_neighbors=30)
neigh.fit(em, lab1)

_,ind = neigh.kneighbors(em2[:,:])

ind = ind.ravel()

def f(x):
    return(lab1[x])

ind2 = np.array(list(map(f,ind))).reshape(10000,30)

ind3 = ind2 == lab2[:].reshape(10000,1)

print('The accuracy obtained is ')
print(sum(ind3.mean(axis=1))/10000)
