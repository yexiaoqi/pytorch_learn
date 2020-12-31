import torch

# torch.autograd.backward(tensors,grad_tensors=None,retain_graph=None,create_graph=False,grad_variables=None)
# torch.autograd.grad(outputs,inputs,grad_outputs=None,retain_graph=None,create_graph=False,only_inputs=True,allow_unused=False)

x=torch.tensor([1],requires_grad=True)
with torch.no_grad():
    y=x*2
print(y.requires_grad)
@torch.no_grad()
def doubler(x):
    return x*2
z=doubler(x)
print(z.requires_grad)