from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
import os
import pandas as pd
import skimage.color as col
import torchvision
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Tiny(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir='tiny-imagenet-200', transform=None, loader = pil_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if transform == None :
            transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomVerticalFlip(p=0.5),torchvision.transforms.ToTensor()])
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        # class_dict -> n01443537 : 0 etc
        self.class_dict = {}
        # rev_dict -> 0 : n01443537 etc
        self.rev_dict = {}
        # image dict -> n01443537 : np.array([n01443537_0.JPEG    n01443537_150.JPEG  n01443537_200.JPEG  n01443537_251.JPEG etc]) 
        self.image_dict = {}
        # big_dict -> idx : [img_name, class]
        self.big_dict = {}

        L = []

        for i,j in enumerate(os.listdir(os.path.join(self.root_dir,'train'))):
            self.class_dict[j] = i
            self.rev_dict[i] = j
            self.image_dict[j] = np.array(os.listdir(os.path.join(self.root_dir,'train',j,'images')))
            for k,l in enumerate(os.listdir(os.path.join(self.root_dir,'train',j,'images'))):
                L.append((l,i))

        for i,j in enumerate(L):
            self.big_dict[i] = j


        self.num_classes = 200

    def _sample(self,idx):

        im, im_class = self.big_dict[idx]

        im2 = np.random.choice(self.image_dict[self.rev_dict[im_class]])

        numbers = list(range(0,im_class)) + list(range(im_class+1,200))
        class3 = np.random.choice(numbers)

        im3 = np.random.choice(self.image_dict[self.rev_dict[class3]])

        p1 = os.path.join(self.root_dir,'train',self.rev_dict[im_class],'images',im)
        p2 = os.path.join(self.root_dir,'train',self.rev_dict[im_class],'images',im2)
        p3 = os.path.join(self.root_dir,'train',self.rev_dict[class3],'images',im3)

        return[p1,p2,p3]

    def __len__(self):
        return 200*500

    def __getitem__(self, idx):
        paths = self._sample(idx)
        images = []
        for i in paths :
            temp = self.loader(i)
            if self.transform:
                temp = self.transform(temp)
            images.append(temp)

        return (images[0],images[1],images[2]),2,3