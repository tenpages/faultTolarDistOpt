import numpy as np
import scipy
from itertools import combinations

A = np.array([[1,0],[.8,.5],[.5,.8],[0,1],[-.5,.8],[-.8,.5]])
x_star = np.array([1,1])
N = np.array([0.00281975,-0.00889553,-0.00534664,0.00518008,0.00247871,-0.00331871])

B = np.matmul(A,x_star) + N
f = 1

n_minus_2f_agents = [list(i) for i in list(combinations([0,1,2,3,4,5], 4))]
x_S = []
for agents in n_minus_2f_agents:
    x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[agents]), A[agents])), np.matmul(np.transpose(A[agents]), B[agents]))
    x_S.append(x)

distances = [np.linalg.norm(x_star-x) for x in x_S]
epsilon = np.max(distances)

import torch
dataset = torch.utils.data.TensorDataset(torch.tensor(A).float(), torch.tensor(B).float())
with open("approximationDataset1","wb") as f:
    torch.save(dataset, f)
