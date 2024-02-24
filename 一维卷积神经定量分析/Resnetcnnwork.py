import torch
from torch import nn
import  torch
from    torch import  nn
from    torch.nn import functional as F

class ResBlk(nn.Module): #设计一个残差模块
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()
        self.res_unit=nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(ch_out)
        )
        if ch_out != ch_in: #由于涉及到相加，因此需要把输入的x维度转成输出的维度
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm1d(ch_out)
            )
    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out=self.res_unit(x)
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out)
        
        return out
class NIR(nn.Module):
    def __init__(self):
        super(NIR,self).__init__()
        self.conv5_unit=nn.Sequential(
            nn.Conv1d(1,40,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(40),
            nn.MaxPool1d(kernel_size=2,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv1d(40,80,kernel_size=3,stride=2,padding=0),
            nn.BatchNorm1d(80),
            nn.AvgPool1d(kernel_size=2,stride=1,padding=0),
            nn.ReLU()
        )
        self.res_uni=nn.Sequential(
            ResBlk(80, 160, stride=2),
            ResBlk(160, 320, stride=2),
            )
        self.lastconv=nn.Sequential(
            nn.Conv1d(320,700,kernel_size=2,stride=1,padding=0),
            nn.ReLU()
        )
        self.pool=nn.Sequential(nn.AdaptiveAvgPool1d(1)) 
        self.flatten=nn.Sequential(
            nn.Linear(700,1),
        )
        #求解卷积后的shape

        # tmp = torch.randn(56,1,700)
        # out = self.conv5_unit(tmp)
        # print('conv out:', out.shape)#conv out:torch.Size([56, 5, 172])
        # out=self.res_uni(out)
        # print('resblok后的卷积',out.shape,out[0,:,:])#resblok后的卷积  torch.Size([56, 60, 1])

    def forward(self, x): #把前向过程走完
        batch_size=len(x) #代表传进来的样本数
       # print(batch_size)
         # [b, 3, 32, 32] => [b, 16, 26, 26]
        x=self.conv5_unit(x)
        x=self.res_uni(x)
        x=self.lastconv(x)
        x=self.pool(x)
        # [b, 16*26*26] =>[b,10]
        x = x.view(batch_size,700 ) #全连接层用二维计算
        logis=self.flatten(x)
        return logis.squeeze(-1)

def main():
    net = NIR()
    tmp = torch.randn(56,1,700)
    out = net(tmp)
    print('lenet out:', out.shape)


if __name__ == '__main__':
    main()
