import numpy as np
import torch

with open("wdbc.data","r") as f:
	X = []
	Y = []
	for line in f:
		line = line.split(",")[1:]
		x = []
		for value in line[2:]:
			x.append(float(value))
		X.append(x)
		if line[0]=='B':
			Y.append(0)
		else:
			Y.append(1) 

X = torch.tensor(X)
Y = torch.tensor(Y).float()

r = torch.transpose(X, 0, 1)
# means = np.array([i.mean() for i in r])
# stds = np.array([i.std() for i in r])
s = np.array([np.array((i - i.mean()) / i.std()) for i in r])
X = torch.transpose(torch.tensor(s), 0, 1)

X_train, Y_train = torch.FloatTensor(X[:400]), torch.FloatTensor(Y[:400])
X_eval, Y_eval = torch.FloatTensor(X[400:]), torch.FloatTensor(Y[400:])

dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float().reshape(-1,1))
with open("wdbcDataset","wb") as f:
	torch.save(dataset, f)

dataset = torch.utils.data.TensorDataset(torch.tensor(X_eval).float(), torch.tensor(Y_eval).float().reshape(-1,1))
with open("wdbcDatasetEval","wb") as f:
	torch.save(dataset, f)
