
import torch
a = torch.tensor([[1,2,3],[4,5,6]])
b = torch.tensor([[1,4,6],[3,4,5]])
combined = torch.cat((a.view(-1), b.view(-1)))
unique, counts = combined.unique(return_counts=True,dim=-1)
intersection = unique[counts > 1]
print(intersection)
