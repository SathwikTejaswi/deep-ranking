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

        self.images = os.listdir(os.path.join(self.root_dir))

        self.image_class = np.array(pd.read_csv('val_details.txt', sep='\t')[['mage','class']]).astype('str')
        self.class_dic = {}
        for i in self.image_class :
            self.class_dic[i[0]]=i[1]

    def _sample(self,idx):
        path = os.path.join(self.root_dir,self.images[idx])
        return path,self.class_dic[self.images[idx]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        paths,lab = self._sample(idx)
        temp = self.loader(paths)
        if self.transform:
            temp = self.transform(temp)
        return temp,lab

print('libraries imoprted')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('loading training data...')
# create data loader
Tiny = Tiny('tiny-imagenet-200/val/images')
dataloader = DataLoader(Tiny, batch_size=100)

#create the model
model  = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,2000)
model = model.to(device)
model.load_state_dict(torch.load('modelresnewnew3.ckpt'))
def forward(x):
    x = x.type('torch.FloatTensor').to(device)
    return(model(x))

L = []
embedding = torch.randn(10,2000).type('torch.FloatTensor').to(device)
model.eval()
with torch.no_grad():
    for k,(i,j) in enumerate(dataloader):
        print(k,end='\r')
        temp = forward(i)
        L = L+list(j)
        embedding = torch.cat((embedding, temp), 0)

np.save('test_embeddings',embedding)
np.save('test_lab',L)
