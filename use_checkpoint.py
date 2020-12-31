import os
import torch
import torchvision as tv
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

batch_size=32
epochs=10
WORKERS=2
test_flag=True
ROOT='./data/cifar-10-batched-py'
log_dir='./models/cifar_model.pth'

transform=tv.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_data=tv.datasets.CIFAR10(root=ROOT,train=True,download=True,transform=transform)
test_data=tv.datasets.CIFAR10(root=ROOT,train=False,download=False,transform=transform)

train_load=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=WORKERS)
test_load=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=WORKERS)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,3,padding=1)
        self.conv2=nn.Conv2d(64,128,3,padding=1)
        self.conv3=nn.Conv2d(128,256,3,padding=1)
        self.conv4=nn.Conv2d(256,256,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(256*8*8,1024)
        self.fc2=nn.Linear(1024,256)
        self.fc3=nn.Linear(256,10)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(F.relu(self.conv2(x)))
        x=F.relu(self.conv3(x))
        x=self.pool(F.relu(self.conv4(x)))
        x=x.view(-1,x.size()[1]*x.size()[2]*x.size()[3])
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model=Net().cuda()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

def train(model,train_loader,epoch):
    model.train()
    train_loss=0
    for i,data in enumerate(train_loader,0):
        x,y=data
        x=x.cuda()
        y=y.cuda()
        optimizer.zero_grad()
        y_hat=model(x)
        loss=criterion(y_hat,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss
    loss_mean=train_loss/(i+1)
    print('Train Epoch:{}\t Loss {:.6f}'.format(epoch,loss_mean.item()))

def test(model,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for i,data in enumerate(test_loader,0):
            x,y=data
            x=x.cuda()
            y=y.cuda()
            optimizer.zero_grad()
            y_hat=model(x)
            test_loss+=criterion(y_hat,y).item()
            pred=y_hat.max(1,keepdim=True)[1]
            correct+=pred.eq(y.view_as(pred)).sum().item()
        test_loss/=(i+1)
        print('Test set: Average loss: {:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,correct,len(test_data),100.*correct/len(test_data)))

def main():
    if test_flag:
        checkpoint=torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs=checkpoint['epoch']
        test(model,test_load)
        return

    # for epoch in range(0,epochs):
    #     train(model,train_load,epoch)
    #     test(model,test_load)
    #     state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
    #     torch.save(state,log_dir)

    if os.path.exists(log_dir):
        checkpoint=torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch=checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch=0
        print('无保存模型，从头开始训练')

    epochs=10
    for epoch in range(start_epoch+1,epochs):
        train(model,train_load,epoch)
        test(model,test_load)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state,log_dir)

if __name__=='__main__':
    main()

