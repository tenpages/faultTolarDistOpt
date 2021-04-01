import numpy as np
import scipy
from itertools import combinations

A = np.array([[1,0],[.8,.5],[.5,.8],[0,1],[-.5,.8],[-.8,.5]])
x_star = np.array([1,1])
#N = np.array([0.00281975,-0.00889553,-0.00534664,0.00518008,0.00247871,-0.00331871])
#N = np.array([-0.0892,0.0349,0.0376,0.0033,-0.0858,-0.0615])
N = np.array([-0.03335922, 0.26583983, -0.27033787, 0.26546552, -0.27614966, 0.1260911])

B = np.matmul(A,x_star) + N
f = 1

# n_minus_2f_agents = [list(i) for i in list(combinations([0,1,2,3,4,5], 4))]
# x_S_hat = []
# for agents in n_minus_2f_agents:
#     x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[agents]), A[agents])), np.matmul(np.transpose(A[agents]), B[agents]))
#     x_S_hat.append(x)
#
# n_minus_f_agents = [list(i) for i in list(combinations([0,1,2,3,4,5], 5))]
# x_S = []
# for agents in n_minus_f_agents:
#     x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[agents]), A[agents])), np.matmul(np.transpose(A[agents]), B[agents]))
#     x_S.append(x)

n_minus_f_agents = [list(i) for i in list(combinations([0,1,2,3,4,5], 5))]
x_S = []
distances = []
for agents in n_minus_f_agents:
    x_S = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[agents]), A[agents])), np.matmul(np.transpose(A[agents]), B[agents]))
    n_minus_2f_agents = [list(i) for i in list(combinations(agents, 4))]
    for agents_2f in n_minus_2f_agents:
        x_S_hat = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[agents_2f]), A[agents_2f])), np.matmul(np.transpose(A[agents_2f]), B[agents_2f]))
        distances.append(np.linalg.norm(x_S-x_S_hat))

epsilon = np.max(distances)
honest = [1,2,3,4,5]
x_0 = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[honest]), A[honest])), np.matmul(np.transpose(A[honest]), B[honest]))
print("epsilon=",epsilon)
print("A=",A)
print("B=",B)
print(x_0)

import torch
dataset = torch.utils.data.TensorDataset(torch.tensor(A).float(), torch.tensor(B).float().reshape(-1,1))
with open("approximationDataset5","wb") as f:
    torch.save(dataset, f)

v_top = []
for Ai in A:
    v_top.append(np.matmul(np.transpose(Ai),Ai))

miu = np.max(v_top)

n_minus_f_agents = [list(i) for i in list(combinations([0,1,2,3,4,5], 5))]
v_mins = []
for agents in n_minus_f_agents:
    v_mins.append(np.min(np.linalg.eigvals(np.matmul(np.transpose(A[agents]),A[agents]))))

gamma = np.min(v_mins)/5
