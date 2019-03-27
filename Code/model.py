import torch
import torchvision
import numpy as np
import torch.nn as nn
from data_utils import *
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


print('libraries imoprted')
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('loading_data...')
# create data loader
TINY = Tiny()
DATALOADER = DataLoader(TINY, batch_size=16,shuffle=True,num_workers=10)

print('define hyper parameters and import model')
# Hyper-parameters
NUM_EPOCHS = 6
LEARNING_RATE = 0.0001

#create the model
MODEL  = models.resnet50(pretrained=True)
num_features = MODEL.fc.in_features
MODEL.fc = nn.Linear(num_features,2000)
MODEL = MODEL.to(DEVICE)
MODEL_PATH = None # Once you have trained this model and have a checkpoint, replace None with the path to the checkpoint

try :
    MODEL.load_state_dict(torch.load(MODEL_PATH))
except :
    pass

# Loss and optimizer
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr = 0.0001, momentum=0.9)

def forward(x):
    x = x.type('torch.FloatTensor').to(DEVICE)
    return(MODEL(x))

# For updating learning rate
def update_lr(OPTIMIZER, lr):
    for param_group in OPTIMIZER.param_groups:
        param_group['lr'] = lr

LOSS_TR = []
BIG_L = []

print('begin training')
# Train the model
TOTAL_STEP = len(DATALOADER)
CURR_LR = LEARNING_RATE

print('')
print('')
for epoch in range(NUM_EPOCHS):
    for i, (D, L, IDX) in enumerate(DATALOADER):
        print(i,end='\r')
        #forward pass
        P = forward(D[0])
        Q = forward(D[1])
        R = forward(D[2])
        # compute loss
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = triplet_loss(P,Q,R)
        # Backward and optimize
        OPTIMIZER.zero_grad()
        loss.backward()
        LOSS_TR.append(loss.item())
        OPTIMIZER.step()
        if (i+1) % 100 == 0:
            temp = sum(LOSS_TR)/len(LOSS_TR)
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, NUM_EPOCHS, i+1, TOTAL_STEP, temp))
            BIG_L = BIG_L + LOSS_TR
            LOSS_TR = []

    # Decay learning rate
    if (epoch+1) % 3 == 0:
        CURR_LR /= 1.5
        update_lr(OPTIMIZER, CURR_LR)

    torch.save(MODEL.state_dict(), 'MRS'+str(epoch)+'.ckpt')
    try :
        np.save('loss_file',BIG_L)
    except :
        pass
