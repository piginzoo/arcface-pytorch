import torch

a = torch.Tensor(torch.randn(30, 2))
b = torch.index_select(a, 0, torch.LongTensor([1, 3, 4, 5]))
print(b)

label = torch.LongTensor(torch.randint(0, 9, (32,)))
data = torch.Tensor(torch.randn(32, 2))
print(label)
for i in range(10):
    mask = (label == i).nonzero(as_tuple=True)
    print(i, "=>", mask[0])
    filtered_data = torch.index_select(data, 0, mask[0])
    print(filtered_data)
    print("-------------------")

print("================================================")

import numpy as np
label = np.random.randint(0, 9, (32,))
data = np.random.rand(32, 2)
print(data)
print(label)
for i in range(10):
    mask = (label == i)
    print(i, "=>", label[mask])
    filtered_data = data[mask]
    print(filtered_data)
    print("-------------------")


# python test_torch.py
