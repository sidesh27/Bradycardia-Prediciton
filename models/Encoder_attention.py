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

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

#Helper functions    

def attn_data(x):
    return x[:,:256,:]

def attn_soft(x):
    return x[:, 256:, :]

def mul(x1, x2):
    return(x1*x2)

def convert_sig(x):
    return(0 if x<0.5 else 1)

class Encoder(nn.Module):
    def __init__(self, n_len, n_class, drop = 0.2):
        super().__init__()
        
        self.conv1 = nn.Sequential( nn.Conv1d(1, 128, kernel_size = 3, stride = 1, padding = 0),
                                    nn.InstanceNorm1d(n_len),
                                    nn.PReLU(),
                                    nn.Dropout(p = drop),
                                    nn.MaxPool1d(kernel_size = 2))
        
        self.conv2 = nn.Sequential( nn.Conv1d(128, 256, kernel_size = 5, stride = 1, padding = 0),
                                    nn.InstanceNorm1d(n_len),
                                    nn.PReLU(),
                                    nn.Dropout(p = drop),
                                    nn.MaxPool1d(kernel_size = 2))
        
        self.conv3 = nn.Sequential( nn.Conv1d(256, 512, kernel_size = 3, stride = 1, padding = 0),
                                    nn.InstanceNorm1d(n_len),
                                    nn.PReLU(),
                                    nn.Dropout(p = drop),
                                    nn.MaxPool1d(kernel_size = 2))
        
        self.attention_data = Lambda(attn_data)
        self.attention_soft = Lambda(attn_soft)
        self.Multiply = Lambda(mul)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu    = nn.ReLU()
        self.d1      = nn.Dropout(0.5)
        self.linear1 = nn.Linear(4, 256)
        self.flatten = nn.Flatten()
        self.Inorm1d = nn.InstanceNorm1d(256)
        self.linear2 = nn.Linear(256*256,1024)
        self.linear3 = nn.Linear(1024,2)
        
        
    def forward(self, x):
            
        # 3 conv blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
            
        #output of conv block--->attention mechanism
        x1 = self.attention_data(x)
        x2 = self.attention_soft(x)
            
        x = self.softmax(x2)
        x = x*x1
        
        #Otuput of attention layer--->Perceptron(sigmoid activation)
        x = self.linear1(x)
        x = self.sigmoid(x)
        

        x = self.Inorm1d(x)
            
        #Output of fully connected--->flatten--->softmax
        x = self.flatten(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.d1(x)
        
        out = self.linear3(x)      
            
        return(out)

def train_model(dataloader_train: DataLoader,
          dataloader_test: DataLoader,
          device: str,
          model: nn.Module,
          loss_func: F,
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
            loss = loss_func(out, y.long())e
            epoch_loss += loss.item()
            loss.backward()
            
            optimiser.step()
            
        loss_history.append(epoch_loss)
       
        #Validation/testing
        running_loss = 0
        running_acc  = 0
        model.eval()
        for batch, data in enumerate(dataloader_test):
            x, y = data
            x,y = x.to(device),(y.view(-1)).to(device)
            out = model(x)
            test_acc = ((F.log_softmax(out, dim=1).argmax(dim=1))== y.long()).cpu().detach().numpy().sum()/len(y)*100
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

df = pd.read_csv("finaldfs/ecghrwindow50overlap75.csv")
train = df.groupby('infant_number').apply(lambda group: group[group["brady_number"] <= (group["brady_number"].max())*0.7]).copy()
test = df.groupby('infant_number').apply(lambda group: group[group["brady_number"] > (group["brady_number"].max())*0.7]).copy()

train_X = train[train.columns[5:-2]].to_numpy()
train_Y = train[train.columns[-2]].to_numpy()

test_X = test[test.columns[5:-2]].to_numpy()
test_Y = test[test.columns[-2]].to_numpy()

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
learning_rate = 0.00001
drop = 0.2
epochs = 1000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Encoder(time_steps,num_classes,0.2)
model.to(device)
loss = F.cross_entropy

model, history = train_model(train_dl, test_dl, device, model, loss,epochs, learning_rate, Save=False)

summary(model, (1,5000))

"""Model summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1            [-1, 128, 4996]             768
    InstanceNorm1d-2            [-1, 128, 4996]               0
             PReLU-3            [-1, 128, 4996]               1
           Dropout-4            [-1, 128, 4996]               0
         MaxPool1d-5            [-1, 128, 2498]               0
            Conv1d-6            [-1, 256, 2488]         360,704
    InstanceNorm1d-7            [-1, 256, 2488]               0
             PReLU-8            [-1, 256, 2488]               1
           Dropout-9            [-1, 256, 2488]               0
        MaxPool1d-10            [-1, 256, 1244]               0
           Conv1d-11            [-1, 512, 1224]       2,753,024
   InstanceNorm1d-12            [-1, 512, 1224]               0
            PReLU-13            [-1, 512, 1224]               1
          Dropout-14            [-1, 512, 1224]               0
        MaxPool1d-15             [-1, 512, 612]               0
           Lambda-16             [-1, 256, 612]               0
           Lambda-17             [-1, 256, 612]               0
          Softmax-18             [-1, 256, 612]               0
           Linear-19             [-1, 256, 256]         156,928
          Sigmoid-20             [-1, 256, 256]               0
   InstanceNorm1d-21             [-1, 256, 256]               0
          Flatten-22                [-1, 65536]               0
           Linear-23                    [-1, 2]         131,074
================================================================
Total params: 3,402,501
Trainable params: 3,402,501
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 70.92
Params size (MB): 12.98
Estimated Total Size (MB): 83.92
----------------------------------------------------------------
"""