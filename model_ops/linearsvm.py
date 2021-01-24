import torch
from torch import nn
from torch.autograd import Variable

import numpy as np
import pandas as pd

import mpi4py as MPI

from model_ops.utils import err_simulation
from compress_gradient import compress


class LinearSVM(nn.Module):
    def __init__(self, size):
        super(LinearSVM, self).__init__()
        self.w = nn.Parameter(torch.randn(1,size), requires_grad = True)
        self.b = nn.Parameter(torch.randn(1), requires_grad = True)

    def forward(self, x):
        h = x.matmul(self.w.t()) + self.b
        return h

    def name(self):
        return 'linearSVM'


class LinearSVM_Split(nn.Module):
    def __init__(self, size):
        super(LinearSVM_Split, self).__init__()
        self.w = nn.Parameter(torch.randn(1,2), requires_grad = True)
        self.b = nn.Parameter(torch.randn(1), requires_grad = True)

    def init_constant(self, value):
        nn.init.constant_(list(self.w.parameters())[0], value)
        nn.init.constant_(list(self.b.parameters())[0], value)

    def forward(self, x):
        h = x.matmul(self.w.t()) + self.b
        return h
