import torch
import Resnetcnnwork
import datalode
import torchvision
import pandas as pd
from datalode import Train_data, Test_data
import torchvision.transforms as transforms
from    torch import nn, optim
import numpy as np
from    torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
#区间缩放，返回值为缩放到[0, 1]区间的数据
device = torch.device('cuda:0')
model =Resnetcnnwork.NIR().to(device)
model.load_state_dict(torch.load('ResnetCifar10.pt'))
#print(model)
#利用hook钩子构建模型
furture_map=[]
input_date=[]
def forword_hook(module,inputdata,output_data):
    furture_map.append(output_data)
    input_date.append(inputdata)
model.lastconv.register_forward_hook(forword_hook)
yy=Test_data()
m,y=yy[0]
y=y.unsqueeze(0)
m=m.unsqueeze(0)
m=m.to(device)
output=model(m)
#print(furture_map[0].shape) torch.Size([1, 700, 42])
weights = model._modules.get('flatten')[0].weight.data   # 获取类别对应的权重
#print(weights.shape)torch.Size([1, 700])
weights=weights.squeeze(0)
weights=weights.view(700, 1)
print(weights.shape)
furture= F.interpolate(furture_map[0], size=(700), mode='linear')
furture=furture.squeeze(0)
print(furture.shape)
print(weights.shape)
cam=weights*furture
print(cam.shape)
cam=torch.sum(cam,0)
cam=cam.unsqueeze(1)
print(cam.shape)
cam=cam.cpu().detach().numpy()
cam=MinMaxScaler(feature_range=(0, 0.3)).fit_transform(cam)
#print(np.count_nonzero(cam >1.1))
#cam[cam<1.1] =0
#print(cam)
np.savetxt('001',cam)
yy=Test_data()
print(yy)

m,y=yy[0]
xx=m.squeeze(0)
plt.figure(500)
print(xx.shape)
x_col1 = np.linspace(0,len(xx),len(xx))  #数组逆序
y_col1 = np.transpose(xx)
plt.plot(x_col1, y_col1)
x_col = np.linspace(0,len(cam[:,0]),len(cam[:,0]))  #数组逆序
x_col=x_col
y_col = (cam[:,0])
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the corn dataset",fontweight= "semibold",fontsize='x-large')
plt.savefig('.MSC111.png')
plt.show()
'''
    x=0
    model_children = list(model.children())
    #print(model_children)
    #print(model_children[0][0])
    #print(len(model_children[1][1].children()))
    #用来弄懂结构图的
    for child in model_children[1][1].children():
        if type(child)==nn.Sequential:
            for mm in child:
                if type(mm)==nn.Conv1d:
                    x+=1
        print(child,x)

    # Load the ResNet-50 Model
    model_weights = []  # save the conv layer weights
    conv_layers = []  # save the 49 conv layers
    # counter: keep count of the conv layers
    counter = 0
'''
'''
# 将所有卷积层以及相应权重加入到两个空list中
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv1d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            if type(model_children[i][j])==nn.Conv1d:
                counter += 1
                model_weights.append(model_children[i][j].weight)
                conv_layers.append(model_children[i][j])
            else:
                for child in model_children[i][j].children():
                    if type(child)==nn.Sequential:
                        for mm in child:
                            if type(mm)==nn.Conv1d:
                                counter += 1
                                model_weights.append(mm.weight)
                                conv_layers.append(mm)
                    #for m in range(len) 
                #if type(child) == nn.Conv1d:
print(f'Total convolutional layers: {counter}')
for weight, conv in zip(model_weights, conv_layers):
    print(f'CONV: {conv} ====> SHAPE: {weight.shape}')
#print(model_weights[1].shape)
'''
'''
# 可视化卷积核图像
plt.figure(figsize=(5, 5))
for i, filter in enumerate(model_weights[0]):
    #print(i,filter.shape)
    plt.subplot(20, 20, i+1)  # conv0: 卷积核大小7*7，共有64个
    plt.imshow(filter[:, :].detach().cpu(), cmap='bone')
    plt.axis('off')
    #plt.savefig('../outputs/filter.png')
#plt.show()

# 可视化图片
yy=Test_data()
m,y=yy[0]
xx=m.squeeze(0)
plt.figure(500)
x_col = np.linspace(0,len(xx),len(xx))  #数组逆序
y_col = np.transpose(xx)
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the corn dataset",fontweight= "semibold",fontsize='x-large')
plt.savefig('.MSC.png')
#plt.show()

#对数据进行变换送进网络
m=m.unsqueeze(0)
m=m.to(device)
#print(m.shape)torch.Size([1, 1, 700])
#pass the image through all the layers生成每个卷积层的特征图
#conv_layers=[conv_layers[0:4],conv_layers[5:10],conv_layers[11:12]]
results = [conv_layers[0](m) ] # conv_layers[0]是卷积层网络结构，后面加入图片才会生成特征图
#print(results.type())
#print(conv_layers[0](m).shape)
#print(conv_layers[1])
#results1=conv_layers[1](results)

for i in range(1, 2):
    results.append(conv_layers[i](results[-1]))  # 每次利用之前生成的最后一张特征图作为输入

# results2=conv_layers[2](results1)
# results3=conv_layers[3](results2)
# results4=conv_layers[4](results1)
#results5=[conv_layers[5](results4+results3)]
#for m in range(3,4):
outputs = results
pic = outputs[0][0,8,:]
print(pic.shape)
x_col = np.linspace(0,len(pic),len(pic))  #数组逆序
#pic=F.relu(pic)
y_col = np.transpose(pic.detach().cpu().numpy())
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the corn dataset",fontweight= "semibold",fontsize='x-large')
plt.savefig('.MSC2.png')
#由于其中有resnet的卷积层 即下面这个卷积层 因此
# if ch_out != ch_in: #由于涉及到相加，因此需要把输入的x维度转成输出的维度
#             # [b, ch_in, h, w] => [b, ch_out, h, w]
#             self.extra = nn.Sequential(
#                 nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=stride),
#                 nn.BatchNorm1d(ch_out)
#             )


# for num_layer in range(len(outputs)):
#     print(num_layer)
#     plt.figure(figsize=(30, 30))
#     layer_viz = outputs[num_layer][0, :, :]
#     layer_viz = layer_viz.data
#     print(layer_viz.size())
#     x_col = np.linspace(0,len(xx),len(xx))  #数组逆序
#     y_col = np.transpose()
#     plt.plot(x_col, y_col)
#     plt.xlabel("Wavenumber(nm)")
#     plt.ylabel("Absorbance")
#     plt.title("The spectrum of the corn dataset",fontweight= "semibold",fontsize='x-large')
#     plt.savefig('.MSC.png')


#     for i, filter in enumerate(layer_viz):
#         if i == 64:  # 只生成64张特征图
#             break
#         plt.subplot(8, 8, i+1)
#         plt.imshow(filter, cmap='gray')
#         plt.axis('off')
#     print(f'Saving layer {num_layer} feature maps ... ')
#     plt.savefig(f'../outputs/layer_{num_layer}.png')
#     plt.close()

# Visualize 64 feature maps from each layer





device = torch.device('cuda:0')
model1 = torch.load('full_weghit')
# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model1.children())
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    for j in range(len(model_children[i])):
        for child in model_children[i][j].children():
            print(child)
            if type(child) == nn.Sequential:
                if type(child)==nn.Conv1d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(len(model_children))
print(model_weights)
print(f"Total convolution layers: {counter}")
print("conv_layers")
model = model1.to(device)
'''