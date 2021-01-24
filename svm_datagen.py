import torch
from sklearn.datasets import make_blobs
import numpy as np

X, Y = make_blobs(n_samples=1200, centers=2, cluster_std=1.2, random_state=1)
X = (X - X.mean()) / X.std()
Y[np.where(Y == 0)] = -1
X_train, Y_train = torch.FloatTensor(X[:1000]), torch.FloatTensor(Y[:1000])
X_eval, Y_eval = torch.FloatTensor(X[1000:]), torch.FloatTensor(Y[1000:])

dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float().reshape(-1,1))
with open("svmDataset","wb") as f:
    torch.save(dataset, f)
dataset = torch.utils.data.TensorDataset(torch.tensor(X_eval).float(), torch.tensor(Y_eval).float().reshape(-1,1))
with open("svmDatasetEval","wb") as f:
    torch.save(dataset, f)
