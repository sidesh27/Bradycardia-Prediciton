import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
from torchsummary import summary
import getopt, sys
import sklearn.metrics as skm

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

#Helper functions    

def convert_sig(x):
    return(0 if x<0.5 else 1)

def pshape(x):
    print(x.shape)

class Encoder(nn.Module):
    def __init__(self,n_filter,n_len, n_class, drop = 0.2):
        super().__init__()
        
        self.n_len = n_len
        self.n_class = n_class
        self.n_filter = n_filter

        self.conv1 = nn.Conv1d(1,n_filter,3,1,0)
        self.bn1   = nn.BatchNorm1d(n_filter)

        self.conv2 = nn.Conv1d(n_filter, 2*n_filter,3,1,0)
        self.bn2   = nn.BatchNorm1d(2*n_filter)

        self.conv3 = nn.Conv1d(2*n_filter,4*n_filter,3,1,0)
        self.bn3   = nn.BatchNorm1d(4*n_filter)
        
        self.conv4 = nn.Conv1d(4*n_filter,8*n_filter,3,1,0)
        self.bn4   = nn.BatchNorm1d(8*n_filter)
        
        self.conv5 = nn.Conv1d(8*n_filter,16*n_filter,3,1,0)
        self.bn5   = nn.BatchNorm1d(16*n_filter)
        
        self.conv6 = nn.Conv1d(16*n_filter,32*n_filter,3,1,0)
        self.bn6   = nn.BatchNorm1d(32*n_filter)
        
        self.fc4   = nn.Linear(512*154,self.n_class)
        self.max   = nn.MaxPool1d(2)
        
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
    
    def forward(self, x):


        x = F.relu(self.bn1(self.conv1(x)))

        x = self.max(x)

        x = F.relu(self.bn2(self.conv2(x)))

        x = self.max(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.max(x)

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.max(x)

        x = F.relu(self.bn5(self.conv5(x)))

        x = self.max(x)

        x = self.flatten(x)

        x = self.fc4(x)
        x = self.sigmoid(x)

        print(x)

        return (x)
            

def train_model(dataloader_train: DataLoader,
          dataloader_test: DataLoader,
          device: str,
          model: nn.Module,
          loss_func: nn,
          epochs: int,
          learning_rate: float,
          Save: bool):

    optimiser = optim.SGD(model.parameters(),lr=learning_rate)
    history = []
    
    loss_history = []
    acc_history = []

    epoch_bar = trange(epochs)
    for epoch in epoch_bar:
    
        epoch_loss = 0
        model.train()
        for batch,data in enumerate(dataloader_train):
            x,y = data
            x,y = x.to(device),(y.view(-1)).to(device)
            
            
            #computation graph (forward prop ->compute loss ->back prop ->update weights)
            optimiser.zero_grad()
            
            out = model(x)
            loss = loss_func(out.view(-1), y)
            epoch_loss += loss.item()
            loss.backward()
            
            optimiser.step()
            
        loss_history.append(epoch_loss)
       
        #Validation/testing
        running_loss = 0
        pred_c0 = 0
        pred_c1 = 0
        gt_c0 = 0
        gt_c1 = 1
        running_acc  = 0
        model.eval()
        for batch, data in enumerate(dataloader_test):
            x, y = data
            x,y = x.to(device),(y.view(-1)).to(device)
            out = model(x)

            
            convert_soft = np.vectorize(convert_sig)
            out = torch.Tensor(convert_soft(out.cpu().detach().numpy())).view(-1).to(device)
            test_acc = (out== y).cpu().detach().numpy().sum()/len(y)*100
            
            pred_c0 = (out == 0).cpu().detach().numpy().sum()
            pred_c1 = (out == 1).cpu().detach().numpy().sum()
            gt_c0   = (y == 0).cpu().detach().numpy().sum()
            gt_c1   = (y == 1).cpu().detach().numpy().sum()
            f1      = skm.f1_score(y.cpu().detach().numpy(), out.cpu().detach().numpy())
            acc_score = skm.accuracy_score(y.cpu().detach().numpy(),out.cpu().detach().numpy())
            
            print("pred 0 :{}  pred 1 :{}  y 0 :{}  y 1 :{}  test_acc: {}  f1: {} acc: {} ".format(pred_c0,pred_c1,gt_c0,gt_c1,test_acc,f1,acc_score))
            
            test_loss = loss_func(out,y).item()
            running_acc  += test_acc*x.size(0)
            running_loss += test_loss*x.size(0)

        test_size = len(dataloader_test.dataset)
        test_acc = running_acc/test_size
        test_loss = running_loss/test_size
        epoch_bar.set_description('acc={0:.2f}%\tBCE={1:.4f}'
                                  .format(test_acc, test_loss))

        history.append((test_acc,test_loss))

    if Save:
        #save
        pass

    return model,history        

df = pd.read_csv("3min_250.csv")
train = df.groupby('infant_no').apply(lambda group: group[group["brady_no"] <= (group["brady_no"].max())*0.8]).copy()
test = df.groupby('infant_no').apply(lambda group: group[group["brady_no"] > (group["brady_no"].max())*0.8]).copy()

train_X = train[train.columns[5:-1]].to_numpy()
train_Y = train[train.columns[-1]].to_numpy()

test_X = test[test.columns[5:-1]].to_numpy()
test_Y = test[test.columns[-1]].to_numpy()

train_X = np.reshape(train_X,(train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X,(test_X.shape[0], 1, test_X.shape[1]))

train_X = torch.Tensor(train_X) # transform to torch tensor
train_Y = torch.Tensor(train_Y)

test_X = torch.Tensor(test_X) # transform to torch tensor
test_Y = torch.Tensor(test_Y)

train_ds = TensorDataset(train_X,train_Y) # create your datset
test_ds = TensorDataset(test_X,test_Y)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl= DataLoader(test_ds, batch_size=32, shuffle=True)

train_size = train_X.shape[0]
test_size = test_X.shape[0]
time_steps = train_X.shape[-1]
num_classes = 1
learning_rate = 0.01
drop = 0.2
epochs = 1
filter_size = 32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Encoder(filter_size,time_steps,num_classes,0.2)
model.to(device)
loss = nn.BCELoss()

model, history = train_model(train_dl, test_dl, device, model, loss,epochs, learning_rate, Save=False)

summary(model, (1,5000))
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1             [-1, 32, 4998]             128
       BatchNorm1d-2             [-1, 32, 4998]              64
         MaxPool1d-3             [-1, 32, 2499]               0
            Conv1d-4             [-1, 64, 2497]           6,208
       BatchNorm1d-5             [-1, 64, 2497]             128
         MaxPool1d-6             [-1, 64, 1248]               0
            Conv1d-7            [-1, 128, 1246]          24,704
       BatchNorm1d-8            [-1, 128, 1246]             256
         MaxPool1d-9             [-1, 128, 623]               0
           Conv1d-10             [-1, 256, 621]          98,560
      BatchNorm1d-11             [-1, 256, 621]             512
        MaxPool1d-12             [-1, 256, 310]               0
           Conv1d-13             [-1, 512, 308]         393,728
      BatchNorm1d-14             [-1, 512, 308]           1,024
        MaxPool1d-15             [-1, 512, 154]               0
          Flatten-16                [-1, 78848]               0
           Linear-17                    [-1, 1]          78,849
          Sigmoid-18                    [-1, 1]               0
================================================================
Total params: 604,161
Trainable params: 604,161
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 15.78
Params size (MB): 2.30
Estimated Total Size (MB): 18.10
----------------------------------------------------------------
"""