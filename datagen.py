import numpy as np
import torch

weight = np.random.rand(1000)*2-1

for i in range(100):
	Xi = np.random.rand(100,1000)-.5
	zi = np.random.rand(100)/5-.1
	Yi = np.matmul(Xi,weight) + zi
	if i==0:
		X = Xi
		Y = Yi
	else:
		X = np.vstack([X,Xi])
		Y = np.append(Y,Yi)

X = torch.tensor(X)
Y = torch.tensor(Y)
dataset = torch.utils.data.TensorDataset(X,Y)
with open("linRegDataset","wb") as f:
	torch.save(dataset, f)
with open("linRegWeight","wb") as f:
	np.save(f, weight)
