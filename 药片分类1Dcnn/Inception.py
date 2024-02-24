import torch
from torch import nn
import  torch
from    torch import  nn
from    torch.nn import functional as F
            
class NIR1(nn.Module):
    def __init__(self):
        super(NIR1,self).__init__()
        self.conv1_unit=nn.Sequential(
            nn.Conv1d(1,8,kernel_size=7,stride=3,padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(),#激活函数
        )
        self.conv2_1unit=nn.Sequential(
            nn.Conv1d(8,12,kernel_size=1,stride=3,padding=0),
            nn.BatchNorm1d(12),
            nn.ReLU(),#激活函数
        )
        self.conv2_2unit=nn.Sequential(
            nn.Conv1d(8,12,kernel_size=3,stride=3,padding=1),
            nn.BatchNorm1d(12),
            nn.ReLU(),#激活函数
        )
        self.conv2_3unit=nn.Sequential(
            nn.Conv1d(8,12,kernel_size=5,stride=3,padding=2),
            nn.BatchNorm1d(12),
            nn.ReLU(),#激活函数
        )
        self.conv3unit=nn.Sequential(
            nn.Conv1d(36,42,kernel_size=3,stride=4,padding=0),
            nn.BatchNorm1d(42),
            nn.ReLU(),#激活函数
        )        
                
        self.flatten=nn.Sequential(
            nn.Linear(42*11,100),
            nn.ReLU(),#激活函数
            nn.Dropout(0.6),
            nn.Linear(100,4),#类别为3
        )
        #tmp = torch.randn(40,36,45)
        # out_n=self.conv1_unit(tmp)#torch.Size([40, 8, 133])第一层
        # out_n=self.conv3unit(tmp)#torch.Size([40, 42, 11, 11])  第三层
        # print(out_n.shape)
        # out_n= self.conv2_1unit(out_n)#torch.Size([40, 12, 45])第2.1层
        # out_n= self.conv2_2unit(out_n)#torch.Size([40, 12, 45]))第2.2层
        # out_n= self.conv2_3unit(out_n)#torch.Size([40, 12, 45])第2.3层
        # print(out_n.shape)
    
    def forward(self, x): #把前向过程走完,传进来
        batch_size=len(x) #代表传进来的样本数        
        x=self.conv1_unit(x)
        m1=self.conv2_1unit(x)
        m2=self.conv2_2unit(x)
        m3=self.conv2_3unit(x)
        x=torch.cat((m1,m2,m3),dim=1)
        x=self.conv3unit(x)
        x=x.view(batch_size,42*11) #展平拉伸
        logis=self.flatten(x)
        return logis

def main():
    net = NIR1()
    tmp = torch.randn(40,1,404)
    out = net(tmp)
    print('lenet out:', out.shape)

if __name__ == '__main__':
    main()