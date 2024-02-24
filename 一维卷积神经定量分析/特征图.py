import torch
import Resnetcnnwork
import datalode
import torchvision
import pandas as pd
from datalode import Train_data, Test_data
import torchvision.transforms as transforms
from    torch import nn, optim
import numpy as np
from    torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
#区间缩放，返回值为缩放到[0, 1]区间的数据
device = torch.device('cuda:0')
model =Resnetcnnwork.NIR().to(device)
model.load_state_dict(torch.load('ResnetCifar10.pt'))
print(model)
#利用hook钩子构建模型
furture_map=[]
input_date=[]
def forword_hook(module,inputdata,output_data):
    furture_map.append(output_data)
    input_date.append(inputdata)
model.lastconv[0].register_forward_hook(forword_hook)
yy=Test_data()
m,y=yy[0]
y=y.unsqueeze(0)
m=m.unsqueeze(0)
m=m.to(device)
output=model(m)
furture= F.interpolate(furture_map[0], size=(700), mode='linear')
furture=F.relu(furture)
print(furture.shape)
map=furture[0].squeeze(0)
map=map.cpu().detach().numpy()
yy=Test_data()
m,y=yy[0]
xx=m.squeeze(0)
plt.figure(500)
print(xx.shape)
x_col = np.linspace(0,len(xx),len(xx))  #数组逆序
y_col = np.transpose(xx)
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the corn dataset",fontweight= "semibold",fontsize='x-large')
plt.savefig('.MSC.png')
for i in  range(0,32):
    x_map=map[i,:]
    x_col1 = np.linspace(0,len(x_map),len(x_map))  #数组逆序
    y_col1 = np.transpose(x_map)
    plt.plot(x_col1, y_col1)
    plt.savefig('.MSC2.png')
plt.show()
