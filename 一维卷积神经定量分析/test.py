import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
from  Resnetcnnwork import NIR
#from    cnn_network import NIR
from datalode import Train_data, Test_data
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda:0')
model = NIR().to(device)
model.load_state_dict(torch.load('ResnetCifar10.pt'))
model.eval()
batch_loss_test = []
yy=Test_data()
test_data=data.DataLoader(yy,batch_size=60,shuffle=True,num_workers=0)
criteon = nn.MSELoss()
correct = 0
for i,y in enumerate(test_data):
    x,label=y
    x,label=x.to(device),label.to(device)
    logits = model(x)#通过网络传回来的值
    test_loss= criteon(logits, label)
    batch_loss_test.append(test_loss.cpu().detach().numpy())  

print('损失率',np.mean(batch_loss_test))