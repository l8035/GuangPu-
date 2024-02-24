import torch
from torch import nn
import  torch
from    torch import  nn
from    torch.nn import functional as F

class NIR(nn.Module):
    def __init__(self):
        super(NIR,self).__init__()
        self.conv3_unit=nn.Sequential(
            nn.Conv1d(1,8,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(),#激活函数
            nn.Conv1d(8,16,kernel_size=3,stride=3,padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),#激活函数
            nn.Conv1d(16,32,kernel_size=5,stride=3,padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),#激活函数
            nn.Conv1d(32,42,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(42),
            nn.ReLU(),#激活函数
        )
        self.flatten=nn.Sequential(
            nn.Linear(42*10,100),
            nn.ReLU(),#激活函数
            nn.Dropout(0.5),
            nn.Linear(100,4),#类别为3
        )

        # tmp = torch.randn(56,1,404)
        # out = self.conv3_unit(tmp)
        # print('conv out:', out.shape)#conv out: torch.Size([56, 2, 84])
    
    def forward(self, x): #把前向过程走完
        batch_size=len(x) #代表传进来的样本数        
        x=self.conv3_unit(x)
        x=x.view(batch_size,42*10 ) #代表传进来的样本数
        logis=self.flatten(x)
        #print(logis.shape)
        return logis

def main():
    net = NIR()
    tmp = torch.randn(56,1,404)
    out = net(tmp)
    print('lenet out:', out.shape)


if __name__ == '__main__':
    main()