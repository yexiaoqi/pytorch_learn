import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

TIME_STEP=10
INPUT_SIZE=1
LR=0.01
DOWNLOAD_MNIST=False

steps=np.linspace(0,np.pi*2,100,dtype=np.float32)
x_np=np.sin(steps)
y_np=np.cos(steps)
plt.plot(steps,y_np,'r-',label='target (cos)')
plt.plot(steps,x_np,'b-',label='input (sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out=nn.Linear(32,1)

    def forward(self,x,h_state):
        r_out,h_state=self.rnn(x,h_state)

        # outs=[]
        # for time_step in range(r_out.size(1)):
        #     outs.append(self.out(r_out[:,time_step,:]))
        # return torch.stack(outs,dim=1),h_state

        r_out=r_out.view(-1,32)
        outs=self.out(r_out)
        return outs.view(-1, TIME_STEP, 1),h_state



rnn=RNN()
print(rnn)


optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.MSELoss()

h_state=None

plt.figure(1,figsize=(12,5))
plt.ion()

for step in range(100):
    start,end=step*np.pi,(step+1)*np.pi
    steps=np.linspace(start,end,10,dtype=np.float32)
    x_np=np.sin(steps)
    y_np=np.cos(steps)

    x=torch.from_numpy(x_np[np.newaxis,:,np.newaxis])
    y=torch.from_numpy(y_np[np.newaxis,:,np.newaxis])

    prediciton,h_state=rnn(x,h_state)
    h_state=h_state.data

    loss=loss_func(prediciton,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps,y_np.flatten(),'r-')
    plt.plot(steps,prediciton.data.numpy().flatten(),'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()
