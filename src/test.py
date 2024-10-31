import torch

a=torch.randn(41,128)
b=torch.randn(28,128)
c=torch.stack([a,b],dim=0)
print(c.shape)
