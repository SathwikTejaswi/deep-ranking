import numpy as np
from sklearn.neighbors import NearestNeighbors

print('libraries imoprted')
print('loading train and test data embeddings')

em = np.load('train_embeddings.npy')[10:,:]

lab1 = np.load('train_lab.npy')

print('fitting NN')
neigh = NearestNeighbors(n_neighbors=30)
neigh.fit(em, lab1)

_,ind = neigh.kneighbors(em[:,:])

ind = ind.ravel()

def f(x):
    return(lab1[x])

ind2 = np.array(list(map(f,ind))).reshape(100000,30)

ind3 = ind2 == lab2[:].reshape(100000,1)

print('The accuracy obtained is ')
print(sum(ind3.mean(axis=1))/100000)