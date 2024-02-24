import numpy as np
import torch
from sklearn.preprocessing import scale,MinMaxScaler,Normalizer,StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torch.utils import data
class PUtao(data.Dataset):
    def __init__(self,shuffle,mode):#root表示文件夹路径
        super(PUtao, self).__init__()
        data_path = 'data/FTIR.csv' #原来的数据文件是120 571
        df = pd.read_csv(data_path,header=None) 
        #print(df.shape)
        dummies = pd.get_dummies(df)
        #print(dummies)#shape(120, 574) 在out中看，增加了4列为如out 120行 574列 即总共有120个数据
        #df1 = dummies.to_csv('out.csv')
        X = dummies.values#在frame中看，列表后四列为分类独热码
        #np.savetxt('frame.csv',X,delimiter=",") #frame: 文件 array:存入文件的数组
        #print(X)
        if shuffle=='y':
            X=np.random.permutation(X)
            x_data = np.array(X[:,:-4]) #-20代表类
            x_data = x_data.astype(np.float32)
            y_data = np.array(X[:,-4:])#左边包含  右边不包含
            np.savetxt('x_data.csv',x_data,delimiter =",",fmt ='%f')
            np.savetxt('y_data.csv',y_data,delimiter =",",fmt ='%f')
        elif shuffle=='n':
            x_data=np.loadtxt('x_data.csv',dtype=np.float32,delimiter=',')
            y_data=np.loadtxt('y_data.csv',dtype=np.float32,delimiter=',')
        x_data=MinMaxScaler(feature_range=(0, 1)).fit_transform(x_data)
        #print(x_data.shape)#(120, 570)
        #print(y_data.shape) #(120, 4)
        self.images=[]
        self.labels=[]
        if mode=='train': 
            for i in range(217):
                self.images.append(x_data[i,:])
                self.labels.append(y_data[i,:])
        elif mode=='val':
            for i in range(217,248):
                self.images.append(x_data[i,:])
                self.labels.append(y_data[i,:])
        else:
            for i in range(248,310):
                self.images.append(x_data[i,:])
                self.labels.append(y_data[i,:])
    def __getitem__(self, index):
        self.images=np.array(self.images)
        self.labels=np.array(self.labels)
        x_txt=torch.from_numpy(self.images[index])
        x_txt=x_txt.unsqueeze(0)#添加一层
        y_txt=torch.tensor(self.labels[index])
        y_txt=np.argmax(y_txt)
        return x_txt,y_txt
    def __len__(self):
        return(len(self.images)) #返回样本个数
def main():
    s=PUtao(mode='test',shuffle='n')#数据集放的位置
    x,y = s.__getitem__(14)
    print('sample:', x.shape, y.shape, y)# torch.Size([1, 448, 448])

if __name__ == '__main__':
    main()


