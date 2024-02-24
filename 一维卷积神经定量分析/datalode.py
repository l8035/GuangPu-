import numpy as np
import torch
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torch.utils import data
data_path = './/data//m5.csv' #数据
torch.set_default_tensor_type(torch.FloatTensor)#设置默认tensor类型为float类型
data1 = np.loadtxt(open(data_path, 'rb'), dtype=np.float32, delimiter=',', skiprows=0)
#print(data)
x_data = np.array(data1[:,:-4])#把倒数第4列前得数据取出 左边包含  右边不包含
X = x_data.astype(np.float32)
x_data=MinMaxScaler(feature_range=(0, 1)).fit_transform(x_data)
y_data = np.array(data1[:,-4])#把Excel表格中的倒数第三例数据取出来
X_train,x_test, y_train,y_test = train_test_split(x_data, y_data, test_size=0.5,random_state=None,shuffle=False) #0.7的训练 ，0.3的测试

class Train_data(data.Dataset):
    def __getitem__(self, index):
        x_txt=torch.from_numpy(X_train[index])
        x_txt=x_txt.unsqueeze(0)#添加一层
        y_txt=torch.tensor(y_train[index])
        return x_txt,y_txt
    def __len__(self):
        return(len(X_train)) #返回样本个数
        
class Test_data(data.Dataset):
    def __getitem__(self, index):
        x_txt=torch.from_numpy(x_test[index])
        y_txt=torch.tensor(y_test[index])
        x_txt=x_txt.unsqueeze(0)#添加一层
        return x_txt,y_txt
    def __len__(self):
        return(len(x_test)) #返回样本个数
# xxx=Test_data()
# x,y=xxx[0]
# x=x.unsqueeze(0)
# print(x.shape,y.shape)

#x=torch.tensor(X_train,dtype=torch.float,requires_grad=True)
#print(X_train[0:8,:,:].shape)
#print(X_train.shape) torch.Size([56, 1, 700])
#绘制原始后图片
'''
plt.figure(500)
x_col = np.linspace(0,len(x_data[0,:]),len(x_data[0,:]))  #数组逆序
y_col = (x_data[30])
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the corn dataset",fontweight= "semibold",fontsize='x-large')
plt.savefig('.MSC.png')
plt.show()

'''
