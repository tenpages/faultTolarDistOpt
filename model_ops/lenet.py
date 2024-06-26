import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from torch.autograd import Variable

from mpi4py import MPI

import sys


class LeNet(nn.Module):
    def __init__(self, channel, size):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(channel, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.size_before_fc = int(((size-4)/2-4)/2)
        self.fc1 = nn.Linear(self.size_before_fc*self.size_before_fc*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, self.size_before_fc*self.size_before_fc*50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def name(self):
        return 'lenet'


class LeNet_Split(nn.Module):
    def __init__(self, channel, size):
        super(LeNet_Split, self).__init__()
        self.conv1 = nn.Conv2d(channel, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.size_before_fc = int(((size-4)/2-4)/2)
        self.fc1 = nn.Linear(self.size_before_fc*self.size_before_fc*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, self.size_before_fc*self.size_before_fc*50)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def name(self):
        return 'lenet_split'
