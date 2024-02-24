import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
import time
from  shallow import NIR
from Resnet import NIR2
from Inception import NIR1
from  datalode import PUtao
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import os
import random
from functools import partial
torch.set_default_tensor_type(torch.FloatTensor)
date_location='You'#或者Putao
log_dir='inception2459+0.925.pt' #1970 对应0.97的损失
batchsz =32
lr = 1e-3
epochs =100000
device = torch.device('cuda')
model = NIR().to(device)
modelname='shallow'
#print(model)
criteon = nn.CrossEntropyLoss()
test_loss = []
batch_loss = []
optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-2)
def val(model, val_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i,y in enumerate(val_loader):
            x,label=y
            x,label=x.to(device),label.to(device)
            logits = model(x)#通过网络传回来的值
            loss=criteon(logits, label)#损失函数
            pred = logits.data.max(1)[1]
            total += label.size(0)
            correct+= pred.eq(label).sum()
            test_loss.append(loss.cpu().detach().numpy())
            #print('label',label.size)
        #print ('coor',correct)
        return  np.mean(test_loss), (correct/total) 
        #print('相对', np.mean(test_loss))
def train(model, train_loader, epoch):
    losses=[]
    model.train()
    for i,y in enumerate(train_loader):#遍历训练集
        x,label=y
        #print(x[0].shape,y[0].shape)
        x,label=x.to(device),label.to(device)
        logits = model(x)#通过网络传回来的值
        loss = criteon(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.cpu().detach().numpy())
        return np.mean(batch_loss)
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, seed):
    worker_seed = seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
def main() :
    seed_everything(11)
    train_loss_show=[]
    test_loss_show=[]
    global test_loss,batch_loss
    train_db =PUtao(mode='train',shuffle='n')
    val_db=PUtao(mode='val',shuffle='n')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                           num_workers=0,worker_init_fn=partial(worker_init_fn,seed=11))
    val_loader=DataLoader(val_db, batch_size=batchsz, shuffle=True,
                           num_workers=0,worker_init_fn=partial(worker_init_fn,seed=11))
    #print(model)
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

    for epoch in range(start_epoch+1, epochs):
            test_loss = []
            batch_loss = []
            trainlo=train(model, train_loader, epoch)
            vall,coor=val(model, val_loader)
            train_loss_show.append(trainlo.item())
            test_loss_show.append(vall.item())
            if trainlo<=0.02:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                torch.save(state,modelname+str(epoch)+'+'+str(round(coor.item(), 3))+'.pt')
                print('epoche:',epoch,'trainlo:',trainlo,'valloss:',vall,'accurate on the testset: %d %%' % (100*coor))
            if coor>=0.87:
                state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                torch.save(state,modelname+str(epoch)+str(round(coor.item(), 3))+'.pt')
                print('epoche:',epoch,'trainlo:',trainlo,'valloss:',vall,'accurate on the testset: %d %%' % (100*coor))
                np.savetxt('numpy_test'+str(epoch)+str(round(coor.item(), 3))+'.csv',  test_loss_show, delimiter =",",fmt ='%f')
                np.savetxt('trainloss'+str(epoch)+str(round(coor.item(), 3))+'.csv',  train_loss_show, delimiter =",",fmt ='%f')                
                break
            if epoch==1000:
                since = time.time()
            if epoch==2000:
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            if  epoch% 1000==0:
                print('epoche:',epoch,'trainlo:',trainlo,'vall:',vall,'accurate on the testset: %d %%' % (100*coor))
                
if __name__ == "__main__":
    main()


