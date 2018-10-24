import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import skimage.color as col
from sklearn.neighbors import NearestNeighbors
from skimage import io, transform
import torchvision.models as models
from joblib import Parallel, delayed
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Tiny(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, loader = pil_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if transform == None :
            transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                        torchvision.transforms.RandomVerticalFlip(p=0.5),
                                                        torchvision.transforms.ToTensor()])
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        # class_dict -> n01443537 : 0 etc
        self.class_dict = {}
        # rev_dict -> 0 : n01443537 etc
        self.rev_dict = {}
        # image dict -> n01443537 : np.array([n01443537_0.JPEG    n01443537_150.JPEG  
        #                               n01443537_200.JPEG  n01443537_251.JPEG etc]) 
        self.image_dict = {}
        # big_dict -> idx : [img_name, class]
        self.big_dict = {}

        L = []

        for i,j in enumerate(os.listdir(os.path.join(self.root_dir))):
            self.class_dict[j] = i
            self.rev_dict[i] = j
            self.image_dict[j] = np.array(os.listdir(os.path.join(self.root_dir,j,'images')))
            for k,l in enumerate(os.listdir(os.path.join(self.root_dir,j,'images'))):
                L.append((l,i))

        for i,j in enumerate(L):
            self.big_dict[i] = j

        self.num_classes = 200

    def _sample(self,idx):
        im, im_class = self.big_dict[idx]
        path = os.path.join(self.root_dir,self.rev_dict[im_class],'images',im)
        return path, im_class

    def __len__(self):
        return len(self.big_dict)

    def __getitem__(self, idx):
        paths,im_class = self._sample(idx)
        temp = self.loader(paths)
        if self.transform:
            temp = self.transform(temp)
        return temp, im_class


print('libraries imoprted')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('loading train and test data embeddings')
# create data loader
Tiny = Tiny('tiny-imagenet-200/train')

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

            