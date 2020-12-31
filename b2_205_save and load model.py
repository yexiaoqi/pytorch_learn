import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TheModelClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        s=self.fc3(x)
        return x

model=TheModelClass()
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name,"\t",optimizer.state_dict()[var_name])

PATH="./models/net.pkl"
torch.save(model.state_dict(),PATH)
model=TheModelClass()
model.load_state_dict(torch.load(PATH))
model.eval()

torch.save({
    'epoch':epoch,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict(),
    'loss':loss,
},PATH)

model=TheModelClass()
optimizer=TheOptimizerClass()

checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch=checkpoint['epoch']
loss=checkpoint['loss']

model.eval()
#model.train()

torch.save({
    'modelA_state_dict':modelA.state_dict(),
    'modelB_state_dict':modelB.state_dict(),
    'optimizerA_state_dict':optimizerA.state_dict(),
    'optimizerB_state_dict':optimizerB.state_dict(),
},PATH)