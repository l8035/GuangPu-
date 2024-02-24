import torch
from torch import nn
import  torch
from    torch import  nn
from    torch.nn import functional as F

class NIR(nn.Module):
    def __init__(self):
        super(NIR,self).__init__()
        self.conv5_unit=nn.Sequential(
            nn.Conv1d(1,10,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(kernel_size=2,stride=1,padding=0),
            nn.Conv1d(10,5,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(5),
            nn.AvgPool1d(kernel_size=2,stride=1,padding=0),
            nn.Conv1d(5,2,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(2),
            nn.AvgPool1d(kernel_size=2,stride=1,padding=0),
        )
        self.flatten=nn.Sequential(
            nn.Linear(84*2,50),
            nn.ReLU(),#激活函数
            nn.Linear(50,25),
            nn.ReLU(),
            nn.Linear(25,1)#后面不再用relu激活函数,因为softmax激活函数包含在损失函数的那个代码里。
        )
       # tmp = torch.randn(56,1,700)
       # out = self.conv5_unit(tmp)
       # print('conv out:', out.shape)#conv out: torch.Size([56, 2, 84])
    def forward(self, x): #把前向过程走完
        batch_size=len(x) #代表的是传进来的图片数量
       # print(batch_size)
         # [b, 3, 32, 32] => [b, 16, 26, 26]
        x=self.conv5_unit(x)
        # [b, 16*26*26] =>[b,10]
        x = x.view(batch_size,84*2 ) #全连接层用二维计算
        logis=self.flatten(x)
        return logis

def main():
    net = NIR()
    tmp = torch.randn(56,1,700)
    out = net(tmp)
    print('lenet out:', out.shape)


if __name__ == '__main__':
    main()