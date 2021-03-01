import torch
from torch import nn
from torch.autograd import Variable

import numpy as np
import pandas as pd

import mpi4py as MPI

from model_ops.utils import err_simulation
from compress_gradient import compress

class LinregTest(nn.Module):
    def __init__(self, size):
        super(LinregTest, self).__init__()
        self.fc1 = nn.Linear(size, 1, bias=False)
        # self.fc2 = nn.Linear(800, 500)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        out = x.view(-1, x.size()[1])
        out = self.fc1(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        # out = self.sigmoid(out)
        return out

    def name(self):
        return 'fc_nn'


class LinregTest_Split(nn.Module):
    def __init__(self, size):
        super(LinregTest_Split, self).__init__()
        self.fc1 = nn.Linear(size, 1, bias=False)
        # self.fc2 = nn.Linear(800, 500)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.full_modules = [self.fc1]
        self._init_channel_index = len(self.full_modules)

    def forward(self, x):
        self.output = []
        self.input = []
        x = x.view(-1, x.size()[1])
        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.fc1(x)
        self.output.append(x)

        return x

    @property
    def fetch_init_channel_index(self):
        return self._init_channel_index

    @property
    def name(self):
        return 'fc_nn'
