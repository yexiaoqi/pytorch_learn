import torch

imgs=torch.randn(1,2,2,3,names=('N','C','H','W'))
print(imgs.names)

imgs.names=['batch','channel','width','height']
print(imgs.names)

imgs=imgs.rename(channel='C',width='W',height='H')
print(imgs.names)

imgs=imgs.rename(None)
print(imgs.names)

unnamed=torch.randn(2,1,3)
print(unnamed)
print(unnamed.names)

imgs=torch.randn(3,1,1,2,names=('N',None,None,None))
print(imgs.names)

imgs=torch.randn(3,1,1,2)
named_imgs=imgs.refine_names('N','C','H','W')
print(named_imgs.names)

def catch_error(fn):
    try:
        fn()
        assert False
    except RuntimeError as err:
        err=str(err)
        if len(err)>180:
            err=err[:180]+"..."
        print(err)

named_imgs=imgs.refine_names('N','C','H','W')
catch_error(lambda :named_imgs.refine_names('N','C','H','Width'))

x=torch.randn(3,names=('X',))
y=torch.randn(3)
z=torch.randn(3,names=('Z',))
catch_error(lambda :x+z)

print((x+y).names)

imgs=torch.randn(2,2,2,2,names=('N','C','H','W'))
per_batch_scale=torch.rand(2,names=('N',))
catch_error(lambda :imgs*per_batch_scale)