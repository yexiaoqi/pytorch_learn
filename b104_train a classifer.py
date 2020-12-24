import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


DOWNLOAD=False

if not(os.path.exists('./data/')) or not os.listdir('./data/'):
    DOWNLOAD=True

transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

trainset=torchvision.datasets.CIFAR10(root='./data/',train=True,download=DOWNLOAD,transform=transform)
trainloader=Data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset=torchvision.datasets.CIFAR10(root='./data/',train=False,download=DOWNLOAD,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter=iter(trainloader)
images,labels=dataiter.next()

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s'%classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

net=Net()
net.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(2):
    running_loss=0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i%2000==1999:
            print('[%d,%5d]loss: %.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss=0.0

print('Finished Training')

PATH='./cifar_net.pth'
torch.save(net.state_dict(),PATH)

# net2=Net()
# net2.load_state_dict(torch.load(PATH))
# outputs=net2(images)
#
# _,predicted=torch.max(outputs,1)
# print('Predicted:',' '.join('%5s'%classes[predicted[j]] for j in range(4)))
#
# correct=0
# total=0
# with torch.no_grad():
#     for data in testloader:
#         images,labels=data
#         outputs=net2(images)
#         _,predicted=torch.max(outputs.data,1)
#         total+=labels.size(0)
#
#         correct+=(predicted==labels).sum().item()
#
# print('Acc of the network on the 10000 test images:%d %%'%(100*correct/total))
#
# class_correct=list(0. for i in range(10))
# print(class_correct)
# class_total=list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images,labels=data
#         outputs=net2(images)
#         _,predicted=torch.max(outputs,1)
#         c=(predicted==labels).squeeze()
#         for i in range(4):
#             label=labels[i]
#             class_correct[label]+=c[i].item()
#             class_total[label]+=1
#
#
#
# for i in range(10):
#     print('acc of %5s :%2d %%'%(classes[i],100*class_correct[i]/class_total[i]))