import numpy as np
import scipy
from itertools import combinations

A = np.array([[1,0],[.8,.5],[.5,.8],[0,1],[-.5,.8],[-.8,.5]])
x_star = np.array([1,1])
#N = np.array([0.00281975,-0.00889553,-0.00534664,0.00518008,0.00247871,-0.00331871])
N = np.array([-0.08921821,0.03487616,0.03758823,0.00334385,-0.08575515,-0.06153888])

B = np.matmul(A,x_star) + N
f = 1

n_minus_2f_agents = [list(i) for i in list(combinations([0,1,2,3,4,5], 4))]
x_S_hat = []
for agents in n_minus_2f_agents:
    x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[agents]), A[agents])), np.matmul(np.transpose(A[agents]), B[agents]))
    x_S_hat.append(x)

n_minus_f_agents = [list(i) for i in list(combinations([0,1,2,3,4,5], 5))]
x_S = []
for agents in n_minus_f_agents:
    x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[agents]), A[agents])), np.matmul(np.transpose(A[agents]), B[agents]))
    x_S.append(x)

distances = []
for i in x_S_hat:
    for j in x_S:
        distances.append(np.linalg.norm(i-j))

epsilon = np.max(distances)
honest = [0,2,3,4,5]
x_0 = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[honest]), A[honest])), np.matmul(np.transpose(A[honest]), B[honest]))
print("epsilon=",epsilon)
print("A=",A)
print("B=",B)
print(x_0)

import torch
dataset = torch.utils.data.TensorDataset(torch.tensor(A).float(), torch.tensor(B).float().reshape(-1,1))
with open("approximationDataset2","wb") as f:
    torch.save(dataset, f)
