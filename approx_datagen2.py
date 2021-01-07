import numpy as np
import scipy
from itertools import combinations

#flag=False
#while (not flag):
#    a=np.random.random([100,10])
#    if np.linalg.matrix_rank(a)==10:
#            flag=True
#
#np.save("matrixA",a)
#N = np.random.uniform(-.1,.1,100)
#np.save("matrixN",N)

A = np.load("matrixA.npy")
x_star = np.array([1]*10)
#N = np.array([0.00281975,-0.00889553,-0.00534664,0.00518008,0.00247871,-0.00331871])
N = np.load("matrixN.npy")

B = np.matmul(A,x_star) + N
np.save("matrixB",B)
f = 2

n_minus_2f_agents = [list(i) for i in list(combinations(list(range(100)), 96))]
x_S_hat = []
for agents in n_minus_2f_agents:
    x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[agents]), A[agents])), np.matmul(np.transpose(A[agents]), B[agents]))
    x_S_hat.append(x)
print("done")

n_minus_f_agents = [list(i) for i in list(combinations(list(range(100)), 98))]
x_S = []
for agents in n_minus_f_agents:
    x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[agents]), A[agents])), np.matmul(np.transpose(A[agents]), B[agents]))
    x_S.append(x)
print("done")

distances = []
count =0
for pos, i in enumerate(x_S_hat):
    if pos>count*len(x_S_hat)/100:
        count+=1
        print('{}%'.format(count))
    for j in x_S:
        distances.append(np.linalg.norm(i-j))
print("done")

epsilon = np.max(distances)
honest = [1,35]
x_0 = np.matmul(np.linalg.inv(np.matmul(np.transpose(A[honest]), A[honest])), np.matmul(np.transpose(A[honest]), B[honest]))
print("epsilon=",epsilon)
print("A=",A)
print("B=",B)
print(x_0)

import torch
dataset = torch.utils.data.TensorDataset(torch.tensor(A).float(), torch.tensor(B).float().reshape(-1,1))
with open("approximationDataset3","wb") as f:
    torch.save(dataset, f)
