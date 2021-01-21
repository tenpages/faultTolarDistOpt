import torch
from sklearn.datasets import make_blobs
import numpy as np

X, Y = make_blobs(n_samples=1000, centers=2, cluster_std=1.2, random_state=1)
X = (X - X.mean()) / X.std()
Y[np.where(Y == 0)] = -1
X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)

dataset = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float().reshape(-1,1))
with open("svmDataset","wb") as f:
    torch.save(dataset, f)
