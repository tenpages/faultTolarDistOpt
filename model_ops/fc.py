import torch
from torch import nn
from torch.autograd import Variable

import numpy as np
import pandas as pd

import mpi4py as MPI

from model_ops.utils import err_simulation
from compress_gradient import compress


class Full_Connected(nn.Module):
    def __init__(self, size):
        super(Full_Connected, self).__init__()
        self.fc1 = nn.Linear(size, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

    def name(self):
        return 'fc_nn'


class Full_Connected_Split(nn.Module):
    def __init__(self, size):
        super(Full_Connected_Split, self).__init__()
        self.fc1 = nn.Linear(size, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

    @property
    def name(self):
        return 'fc_nn'