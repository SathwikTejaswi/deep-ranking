import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import skimage.color as col
from skimage import io, transform
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Test(Dataset):
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

        self.images = os.listdir(os.path.join(self.root_dir))

        self.image_class = np.array(pd.read_csv('val_details.txt', sep='\t')[['mage','class']]).astype('str')
        self.class_dic = {}
        for i in self.image_class :
            self.class_dic[i[0]]=i[1]

    def sample(self,idx):
        return self.images[idx]

    def __len__(self):
        return len(self.images)


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

    def sample(self,idx):
        im, im_class = self.big_dict[idx]
        path = os.path.join(self.root_dir,self.rev_dict[im_class],'images',im)
        return path, im_class

    def __len__(self):
        return len(self.big_dict)


print('libraries imoprted')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('loading training data...')
# create data loader
test = Test('tiny-imagenet-200/val/images')
train = Tiny('tiny-imagenet-200/train')

train_em = np.load('train_embeddings.npy').reshape(100010,2000)[10:,:]
test_em = np.load('test_embeddings.npy').reshape(10010,2000)[10:,:]

print('------------------')
for i in range(5):
    idx = np.random.randint(10000)
    temp = test_em[idx].reshape(1,2000)
    print('QUERY IMAGE :')
    print(test.sample(idx))

    temp = (train_em-temp)**2
    temp = temp.sum(axis=1)
    print ((np.sort(temp)**0.5)[:30])
    temp2 = temp.argsort()[:30]
    print('RESULTS ARE:')
    for j in temp2:
        print(train.sample(j)[0])

    print('------------------')