import numpy as np
import torch

X=[]
for i in range(10):
	t=[]
	for j in range(10):
		if j==i:
			t.append(1)
		else:
			t.append(0)
	X.append([t]*19)

X = np.array(X).reshape(-1,10)
weight = np.array([1]*10)

Y = np.matmul(X, weight).reshape(-1,1) + np.random.rand(190)/50-.01
Yp = np.matmul(X, weight).reshape(-1,1)

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
Yp = torch.tensor(Yp).float()

dataset = torch.utils.data.TensorDataset(X,Y)
dataset2 = torch.utils.data.TensorDataset(X,Yp)

with open("linRegDataset2","wb") as f:
	torch.save(dataset, f)
with open("linRegDataset3","wb") as f:
	torch.save(dataset2, f)
with open("linRegWeight2","wb") as f:
	np.save(f, weight)
