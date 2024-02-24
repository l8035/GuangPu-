import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
from  shallow import NIR
from Resnet import NIR2
from Inception import NIR1
from  datalode import PUtao
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.FloatTensor)
log_dir='inception1231+1.0+0.967.pt' #
device = torch.device('cuda:0')
model = NIR1().to(device)
checkpoint = torch.load(log_dir)
model.load_state_dict(checkpoint['model'])
model.eval()
batch_loss_test = []
test_db = PUtao('n','test')
test_data = DataLoader(test_db, batch_size=62,shuffle=None, num_workers=0)
#print(len(test_data)) 表示怎共有多少轮 
criteon = nn.CrossEntropyLoss()
correct = 0
for i,y in enumerate(test_data):
    x,label=y
    x,label=x.to(device),label.to(device)
    print('lll',label.shape)
    logits = model(x)#通过网络传回来的值
    test_loss= criteon(logits, label)#把传回来的值与实际的标签进行损失运算
    batch_loss_test.append(test_loss.cpu().detach().numpy()) #求解平均损失
    #print(logits.shape)#torch.Size([4, 2])
    pred = logits.data.max(1)[1]#用于寻找最大数据的下标，max(1)[1] 第一个1表示第1个维度，第二个1表示返回最大的只返回最大值的每个索引
    print('预测',pred)
    print('lable',label)
    correct = pred.eq(label).sum()#比较label与索引值是否一致
x=test_data.batch_size
print(x,correct,len(test_data))
mean=(correct)/(x)
print('正确率',mean)
print('损失率',np.mean(batch_loss_test))