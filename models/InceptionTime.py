import pandas as pd
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
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
from tsai.imports import *
from tsai.models.layers import *
from tsai.models.utils import *

def noop(x):
    pass

def shortcut(c_in, c_out):
    return nn.Sequential(*[nn.Conv1d(c_in, c_out, kernel_size=1), 
                           nn.BatchNorm1d(c_out)])
def convert_sig(x):
    return(0 if x<0.5 else 1)

class InceptionModule(Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 0 else False
        self.bottleneck = Conv1d(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList([Conv1d(nf if bottleneck else ni, nf, k, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), Conv1d(ni, nf, 1, bias=False)])
        self.concat = Concat()
        self.bn = BN1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return self.act(self.bn(x))


@delegates(InceptionModule.__init__)
class InceptionBlock(Module):
    def __init__(self, ni, nf=32, residual=True, depth = 6, **kwargs):
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * 4, nf, **kwargs))
            if self.residual and d % 3 == 2: 
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(BN1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out, 1, act=None))
        self.add = Add()
        self.act = nn.ReLU()
        
    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2: res = x = self.act(self.add(x, self.shortcut[d//3](res)))
        return x

    
@delegates(InceptionModule.__init__)
class InceptionTime(Module):
    def __init__(self, c_in, c_out, nf=32, nb_filters=None, **kwargs):
        nf = ifnone(nf, nb_filters) # for compatibility
        self.inceptionblock = InceptionBlock(c_in, nf, **kwargs)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * 4, c_out)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.sig(self.fc(x))
        return x
    
def train_model(train_dl: DataLoader,
          test_dl: DataLoader,
          device: str,
          model: nn.Module,
          epochs: int,
          learning_rate: float,
          Save: bool):

    optimiser = optim.Adam(model.parameters() ,lr=learning_rate)
    history = []
        
    loss_history = []
    acc_history = []

    epoch_bar = trange(epochs)
    for epoch in epoch_bar:
    
        epoch_loss = 0
        model.train(mode = True)
        for batch, data in enumerate(train_dl):
            x,y = data
            x,y = x.to(device),y.to(device)
                 
            #computation graph (forward prop ->compute loss ->back prop ->update weights)
            optimiser.zero_grad()
            
            out = model(x)
            
            y = torch.Tensor(y.cpu().detach().numpy()).view(y.shape[0], 1).to(device)
          
            loss = loss_func(out, y)
            epoch_loss += loss.item()
            loss.backward()
            
            optimiser.step()
            
        loss_history.append(epoch_loss)
        print ("Train Loss: ",epoch_loss/len(train_dl))

        #Validation

        running_loss = 0
        running_acc  = 0
        running_far = 0
        model.eval()
        for batch, data in enumerate(test_dl):
            x, y = data
            x, y = x.to(device),y.to(device)
            out = model(x)
            convert_soft = np.vectorize(convert_sig)
            out1 = torch.Tensor(convert_soft(out.cpu().detach().numpy())).view(-1).to(device)
            test_acc = (out1 == y.view(-1)).cpu().detach().numpy().sum()/len(y)
            y = torch.Tensor(y.cpu().detach().numpy()).view(y.shape[0], 1).to(device)            
            test_loss = loss_func(out,y).item()
            running_acc  += test_acc
            running_loss += test_loss

        test_size = len(test_dl)
        test_acc = running_acc/(batch+1)
        test_loss = running_loss/(batch+1)

        
        epoch_bar.set_description('acc={0:.2f}%\tBCE={1:.4f}%'
                                  .format(test_acc, test_loss))


    return model,history 

if __name__ == "__main__":
    
    arch = InceptionTime(1, 1)
    
    df = pd.read_csv('finaldfs/ecgfiltered30sec.csv', index_col = 0)
    
    train = df.groupby('infant_no').apply(lambda group : group[group['brady_no'] <= (group['brady_no'].max())*0.7]).copy()
    test = df.groupby('infant_no').apply(lambda group : group[group['brady_no'] > (group['brady_no'].max())*0.7]).copy()
    
    
    x_train = train[train.columns[4:-2]]
    y_train = train['brady']
    x_test = test[test.columns[4:-2]]
    y_test = test['brady']
    
    x_train = np.expand_dims(x_train, axis = 1)
    x_test = np.expand_dims(x_test, axis = 1)
        
    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    y_train = torch.Tensor(y_train.to_numpy())
    y_test = torch.Tensor(y_test.to_numpy())
        
    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    
    train_dl = DataLoader(train_ds, batch_size = 10, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = 10, shuffle = True)
    
    device = torch.device('cuda:0')
    
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    time_steps = x_train.shape[-1]
    num_classes = 1
    learning_rate = 1e-6
    drop = 0.2
    epochs = 100
    loss_func = nn.BCELoss()
    
    model = arch
    model = model.to(device)
    train_model(train_dl,
          test_dl,
          device,
          model,
          epochs,
          learning_rate, False)