import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from torch.autograd import Variable

from mpi4py import MPI

import sys


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)
        # loss = self.criterion(x, target)
        return x

    def name(self):
        return 'lenet'


class LeNet_Split(nn.Module):
    def __init__(self):
        super(LeNet_Split, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.criterion = nn.CrossEntropyLoss()

        self.full_modules = [self.conv1,self.conv2,self.fc1, self.fc2]
        self._init_channel_index = len(self.full_modules)*2

    def forward(self, x):
        self.output = []
        self.input = []

        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)
        # loss = self.criterion(x, target)
        return x

    @property
    def fetch_init_channel_index(self):
        return self._init_channel_index

    def name(self):
        return 'lenet_split'
