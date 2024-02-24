import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
from  Resnetcnnwork import NIR
#from    cnn_network import NIR
from datalode import Train_data, Test_data
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
torch.set_default_tensor_type(torch.FloatTensor)
def main() :
    device = torch.device('cuda:0')
    model = NIR().to(device)
    criteon = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print(model)
    losses = []
    XX=Train_data()
    yy=Test_data()
    train_data=data.DataLoader(XX,batch_size=8,shuffle=True,num_workers=0)
    test_data=data.DataLoader(yy,batch_size=8,shuffle=True,num_workers=0)    
    model.train()
    for epoch in range(2000):
        batch_loss = []
        for i,y in enumerate(train_data):
            x,label=y
            x,label=x.to(device),label.to(device)
            logits = model(x)#通过网络传回来的值
            loss = criteon(logits, label)#损失函数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.cpu().detach().numpy())      
        if epoch% 100==0:
            losses.append(np.mean(batch_loss))
            print(epoch, np.mean(batch_loss))
    model.eval()
    test_loss = []
    for i,y in enumerate(test_data):
        x,label=y
        x,label=x.to(device),label.to(device)
        logits = model(x)#通过网络传回来的值
        test_loss.append(loss.cpu().detach().numpy())      
        print('相对', np.mean(test_loss))
    #torch.save(model.state_dict(), './capsnet_ep{}_acc{}.pt'.format(1000, np.mean(test_loss)))
    #测试第一个值与最后一个值的大小比较
    torch.save(model.state_dict(), 'ResnetCifar10.pt')
    # m,y=yy[0]
    # m=m.unsqueeze(0)
    # m=m.to(device)
    # print(m.shape)
    # L=model(m)
    # print(L,y)
    ''''
    plt.figure(500)
    x_col = np.linspace(0,len(predict[0,:]),len(x_data[0,:]))  #数组逆序
    y_col = np.transpose(x_data)
    plt.plot(x_col, y_col)
    plt.xlabel("Wavenumber(nm)")
    plt.ylabel("Absorbance")
    plt.title("The spectrum of the corn dataset",fontweight= "semibold",fontsize='x-large')
    plt.savefig('.MSC.png')
    plt.show()
    '''

if __name__ == "__main__":
    main()

