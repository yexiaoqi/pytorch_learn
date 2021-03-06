import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm

torch.manual_seed(1)

EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST=True

train_data=torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data=torchvision.datasets.MNIST(root='./mnist/',train=False)

train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_x=torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:2000].cuda()/255.
test_y=test_data.test_labels[:2000].cuda()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output
cnn=CNN()
cnn.cuda()
print(cnn)

optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x=x.cuda()
        b_y=y.cuda()

        output=cnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50==0:
            test_output=cnn(test_x)
            pred_y=torch.max(test_output,1)[1].cuda().data
            acc=torch.sum(pred_y==test_y).type(torch.FloatTensor)/float(test_y.size(0))
            print('Epoch: ',epoch,'| train loss: %.4f'%loss.data.cpu().numpy(),'| test acc:%2f'%acc)


test_output=cnn(test_x[:10])
pred_y=torch.max(test_output,1)[1].cuda().data
print(pred_y,'prediction num')
print(test_y[:10],'real')