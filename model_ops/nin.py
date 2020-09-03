import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from torch.autograd import Variable

from mpi4py import MPI

import sys


class NiN(nn.Module):
    def __init__(self, channel, init_weights=True):
        super(NiN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(channel,192,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,160,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160,96,kernel_size=1,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            nn.Dropout(0.5),

            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
        )
        if init_weights:
            self._init()
    
    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), 10)
        return x
    
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
