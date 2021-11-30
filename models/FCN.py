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

class FCN(nn.Module):
    def __init__(self, n_len, n_class, drop = 0.2):
        super().__init__()
        
        self.n_len = n_len
        self.n_class = n_class

        self.conv1 = nn.Conv1d(1,128,8,1,0)
        self.bn1   = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128,256,5,1,0)
        self.bn2   = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256,128,3,1,0)
        self.bn3   = nn.BatchNorm1d(128)

        self.fc4   = nn.Linear(128,self.n_class)
   
    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool1d(x,2)
        x = torch.mean(x,dim=2)
        x = self.fc4(x)

        return (x)

def train_model(dataloader_train: DataLoader,
          dataloader_test: DataLoader,
          device: str,
          model: nn.Module,
          loss_func: F,
          epochs: int,
          learning_rate: float,
          Save: bool):

    optimiser = optim.Adam(model.parameters(),lr=learning_rate)
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
            loss = loss_func(out, y.long())
            epoch_loss += loss.item()
            loss.backward()
            
            optimiser.step()
            
        loss_history.append(epoch_loss)
       
        #Validation

        running_loss = 0
        running_acc  = 0
        model.eval()
        for batch, data in enumerate(dataloader_test):
            x, y = data
            x,y = x.to(device),(y.view(-1)).to(device)
            out = model(x)
            test_acc = ((torch.argmax(out,1))== y.long()).cpu().detach().numpy().sum()/len(y)*100
            test_loss = F.cross_entropy(out,y.long()).item()
            running_acc  += test_acc*x.size(0)
            running_loss += test_loss*x.size(0)

        test_size = len(dataloader_test.dataset)
        test_acc = running_acc/test_size
        test_loss = running_loss/test_size
        epoch_bar.set_description('acc={0:.2f}%\tcross entropy={1:.4f}'
                                  .format(test_acc, test_loss))

        history.append((test_acc,test_loss))

    if Save:
        #save
        pass

    return model,history        

df = pd.read_csv("3min_250.csv")
train = df.groupby('infant_no').apply(lambda group: group[group["brady_no"] <= (group["brady_no"].max())*0.7]).copy()
test = df.groupby('infant_no').apply(lambda group: group[group["brady_no"] > (group["brady_no"].max())*0.7]).copy()

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
num_classes = len(np.unique(train_Y))
learning_rate = 0.001
drop = 0.2
epochs = 1000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = FCN(time_steps,num_classes,0.2)
model.to(device)
loss = F.cross_entropy

model, history = train_model(train_dl, test_dl, device, model, loss,epochs, learning_rate, Save=False)