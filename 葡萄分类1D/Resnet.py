import torch
from torch import nn
import  torch
from    torch import  nn
from    torch.nn import functional as F
class ResBlk1(nn.Module): #设计一个残差模块

    def __init__(self, ch_in=8, ch_out=16, stride=2):
        super(ResBlk1, self).__init__()
        self.conv2_unit1=nn.Sequential(
        nn.Conv1d(8,16,kernel_size=3,stride=2,padding=1),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Conv1d(16,32,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm1d(32),   
        )
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=2,padding=0),
                nn.BatchNorm1d(ch_out),  
            )
        # tmp1 = torch.randn(40,8,49)    #torch.Size([40, 8, 49, 49]) 第一层以后的shape
        # out2_1=self.conv2_unit1(tmp1) #torch.Size([40, 16, 25, 25])
        # print(out2_1.shape)#torch.Size([40, 16, 25, 25])
        # out2_2=self.extra(tmp1)#torch.Size([40, 16, 25, 25])
        # print(out2_2.shape)
    def forward(self, x):
        out=self.conv2_unit1(x)
        out = self.extra(x)+out#torch.Size([40, 70, 25, 25])
        out = F.relu(out)
        return out
class NIR2(nn.Module):
    def __init__(self):
        super(NIR2,self).__init__()
        self.conv1_unit=nn.Sequential(
            nn.Conv1d(1,8,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(),#激活函数
        )#第一层 torch.Size([40, 8, 49, 49])
 
        self.blk1 = ResBlk1(8, 32, stride=2) #因为上一层输入的维度为8
        self.conv2_unit=nn.Sequential(
            nn.Conv1d(32,42,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(42),
            nn.ReLU(),#激活函数
        )#第一层 torch.Size([40, 8, 49, 49])
        self.flatten=nn.Sequential(
            nn.Linear(42*12,100),
            nn.ReLU(),#激活函数
            nn.Dropout(0.5),
            nn.Linear(100,20),
        )
        # tmp = torch.randn(40,1,100)
        # out_n=self.conv1_unit(tmp)# torch.Size([40, 8,  49])
        # out_n= self.blk1(out_n)#torch.Size([40, 32, 25, 25])
        # out=self.conv2_unit(out_n)# torch.Size([40, 42, 12])
        # print('conv out:', 'm.shape',out.shape)
    
    def forward(self, x): #把前向过程走完,传进来
        batch_size=len(x) #代表传进来的样本数        
        x=self.conv1_unit(x)
        x= self.blk1(x)# torch.Size([40, 16, 75, 75])
        x=self.conv2_unit(x)#torch.Size([40, 32, 19, 19])
        x=x.view(batch_size, 42*12) #展平拉伸
        logis=self.flatten(x)
        return logis

def main():
    net = NIR2()
    #net2=ResBlk1(8, 32, stride=2)
    tmp = torch.randn(40,1,100)
    out = net(tmp)
    print('lenet out:', out.shape)

if __name__ == '__main__':
    main()