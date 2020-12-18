import torch
import torch.nn.functional as F

net2=torch.nn.Sequential(torch.nn.Linear(1,10),torch.nn.ReLU(),torch.nn.Linear(10,1))
print(net2)