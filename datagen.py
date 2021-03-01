import numpy as np
import torch
import random

X_=[]
'''
for i in range(10):
    t=[]
    for j in range(10):
        if j==i:
            t.append(1)
        else:
            t.append(random.random())
            # t.append(0)
    X_.append([t]*19)
'''
for i in range(10000):
    t=[]
    for j in range(10):
        if j==i%10:
            t.append(1)
        else:
            t.append(random.random())
            # t.append(0)
    X_.append([t])

X = np.array(X_).reshape(-1,10)
# weight = np.array([.5]*10)
# weight = np.array([i/10 for i in range(1,11)])
weight = np.array([i/10 for i in range(1,10*2,2)])

# Y = np.matmul(X, weight).reshape(-1,1) + np.random.rand(190).reshape(-1,1)/50-.01
Y = np.matmul(X, weight).reshape(-1,1) + np.random.rand(len(X)).reshape(-1,1)/50-.01
Yp = np.matmul(X, weight).reshape(-1,1)

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
Yp = torch.tensor(Yp).float()

dataset = torch.utils.data.TensorDataset(X,Y)
dataset2 = torch.utils.data.TensorDataset(X,Yp)

with open("linRegDataset","wb") as f:
	torch.save(dataset2, f)
