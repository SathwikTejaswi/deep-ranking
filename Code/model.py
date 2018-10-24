import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from data_utils import *
from torch.utils.data import Dataset, DataLoader
import numpy as np

print('libraries imoprted')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('loading_data...')
# create data loader
Tiny = Tiny()
dataloader = DataLoader(Tiny, batch_size=16,shuffle=True,num_workers=10)

print('define hyper parameters and import model')
# Hyper-parameters
num_epochs = 6
learning_rate = 0.0001

#create the model
model  = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,2000)
model = model.to(device)

model.load_state_dict(torch.load('modelresnewnew3.ckpt'))
# Loss and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9)


def forward(x):
    x = x.type('torch.FloatTensor').to(device)
    return(model(x))

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

Loss_tr = []
Big_L = []

print('begin training')
# Train the model
total_step = len(dataloader)
curr_lr = learning_rate

print('')
print('')
for epoch in range(num_epochs):
    for i, (D, L, IDX) in enumerate(dataloader):
        print(i,end='\r')
        #forward pass
        P = forward(D[0])
        Q = forward(D[1])
        R = forward(D[2])

        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss(P,Q,R)


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        Loss_tr.append(loss.item())
        optimizer.step()
        if (i+1) % 100 == 0:
            temp = sum(Loss_tr)/len(Loss_tr)
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, temp))
            Big_L = Big_L + Loss_tr
            Loss_tr = []

    # Decay learning rate
    if (epoch+1) % 3 == 0:
        curr_lr /= 1.5
        update_lr(optimizer, curr_lr)

    torch.save(model.state_dict(), 'MRS'+str(epoch)+'.ckpt')
    try :
        np.save('loss_file',Big_L)
    except :
        pass
