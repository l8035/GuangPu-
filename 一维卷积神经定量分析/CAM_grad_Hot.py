import torch
import Resnetcnnwork
import datalode
import torchvision
from datalode import Train_data, Test_data
import torchvision.transforms as transforms
from    torch import nn, optim
import numpy as np
from    torch.nn import functional as F

device = torch.device('cuda:0')
import matplotlib.pyplot as plt
model =Resnetcnnwork.NIR().to(device)
model.load_state_dict(torch.load('ResnetCifar10.pt'))
lose_func = nn.MSELoss().to(device)
for_mape=[]
input_date=[]
out_grident=[]
input_grindent=[]
def forword_hook(module,inputdata,output_data):
    for_mape.append(output_data)
    input_date.append(inputdata)
def backword_hook(module,inputdata1,output_data1):
    print('哈哈哈',output_data1)
    #print(inputdata1)
print(model.conv5_unit[2])
model.conv5_unit[2].register_forward_hook(forword_hook)
model.conv5_unit[2].register_backward_hook(backword_hook)
yy=Test_data()
m,y=yy[0]
y=y.unsqueeze(0)
m=m.unsqueeze(0)
m=m.to(device)
output=model(m)
loss=lose_func(output,y.to(device))
print(output,y)
loss.backward()
print('output',output.shape)
print('for_map',for_mape[0].shape)
print(input_date[0][0].shape)
    






'''
model_children = list(model.children())
for name, parameters in model.named_parameters():  
    print(name, ';', parameters.size())
pic=model.state_dict()['flatten.0.weight']
pic=pic[0,:]

x_col = np.linspace(0,len(pic),len(pic))  #数组逆序
y_col = np.transpose(pic.detach().cpu().numpy())
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the corn dataset",fontweight= "semibold",fontsize='x-large')
plt.savefig('.MSC2.png')
'''