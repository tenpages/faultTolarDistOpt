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

        self.full_modules = [self.fc1, self.fc2, self.fc3]
        self._init_channel_index = len(self.full_modules)*2

    def forward(self, x):
        """
        The structure of the network:
        fully-connected layer: in: 784, out: 800;
        relu
        fully-connected layer: in: 800, out: 500;
        relu
        fully-connected lyaer: in: 500, out: 10;
        sigmoid
        :param x:
        :return:
        """
        self.output = []
        self.input = []
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.fc1(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.relu(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.fc2(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.relu(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.fc3(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.sigmoid(x)
        self.output.append(x)

        return x

    @property
    def fetch_init_channel_index(self):
        return self._init_channel_index

    def backward_normal(self, g, communicator, req_send_check, cur_step, fail_workers,
                        err_mode, compress_grad):
        """

        :param g:
        :param communicator:
        :param req_send_check:
        :param cur_step:
        :param fail_workers:
        :param err_mode:
        :param compress_grad:
        :return:
        """
        mod_avail_index = len(self.full_modules)-1
        channel_index = self._init_channel_index - 2
        mod_counters_ = [0]*len(self.full_modules)
        for i, output in reversed(list(enumerate(self.output))):
            req_send_check[-1].wait()
            if i==(len(self.output)-1):
                # for the las node
                output.backward(g)
            else:
                output.backward(self.input[i+1].grad.data)
                tmp_grad_weight = self.full_modules[mod_avail_index].weight.grad
                tmp_grad_bias = self.full_modules[mod_avail_index].bias.grad

                if mod_avail_index==len(self.full_modules)-1:
                    if not pd.isnull(tmp_grad_weight):
                        grads=tmp_grad_weight.data.numpy().astype(np.float64)
                        # if communicator.Get_rank() in fail_workers:
                        if compress_grad == 'compress':
                            _compressed_grad=compress(grads)
                            req_isend=communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                        else:
                            req_isend=communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                        req_send_check.append(req_isend)
                        mod_avail_index-=1
                        channel_index-=1
                    else:
                        continue
                else:
                    if not pd.isnull(tmp_grad_weight) and not pd.isnull(tmp_grad_bias):
                        if mod_counters_[mod_avail_index]==0:
                            grads=tmp_grad_bias.data.numpy().astype(np.float64)
                            if compress_grad == 'compress':
                                _compressed_grad=compress(grads)
                                req_isend=communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                            else:
                                req_isend=communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            req_send_check.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                        elif mod_counters_[mod_avail_index]==1:
                            grads=tmp_grad_weight.data.numpy().astype(np.float64)
                            if compress_grad == 'compress':
                                _compressed_grad=compress(grads)
                                req_isend=communicator.isend(_compressed_grad, dest=0, tag=88+channel_index)
                            else:
                                req_isend=communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88+channel_index)
                            req_send_check.append(req_isend)
                            channel_index-=1
                            mod_counters_[mod_avail_index]+=1
                    else:
                        continue

        if mod_counters_[0]==1:
            req_send_check[-1].wait()
            grads=tmp_grad_weight.data.numpy().astype(np.float64)
            if compress_grad == 'compress':
                _compressed_grad = compress(grads)
                req_isend = communicator.isend(_compressed_grad, dest=0, tag=88 + channel_index)
            else:
                req_isend = communicator.Isend([grads, MPI.DOUBLE], dest=0, tag=88 + channel_index)
            req_send_check.append(req_isend)
        return req_send_check

    def backward_coded(self, g, cur_step):
        grad_aggregate_list = []
        mod_avail_index=len(self.full_modules)-1
        channel_index=self._init_channel_index-2
        mod_counters_ = [0]*len(self.full_modules)
        for i,output in reversed(list(enumerate(self.output))):
            if i==(len(self.output)-1):
                output.backward(g)
            else:
                output.backward(self.input[i+1].grad.data)
                tmp_grad_weight=self.full_modules[mod_avail_index].weight.grad
                tmp_grad_bias=self.full_modules[mod_avail_index].bias.grad
                if not pd.isnull(tmp_grad_weight) and not pd.isnull(tmp_grad_bias):
                    if mod_counters_[mod_avail_index]==0:
                        grads=tmp_grad_bias.data.numpy().astype(np.float64)
                        grad_aggregate_list.append(grads)
                        channel_index-=1
                        mod_counters_[mod_avail_index]+=1
                    elif mod_counters_[mod_avail_index]==1:
                        grads=tmp_grad_weight.data.numpy().astype(np.float64)
                        grad_aggregate_list.append(grads)
                        channel_index-=1
                        mod_counters_[mod_avail_index]+=1
                        mod_avail_index-=1
                else:
                    continue

        if mod_counters_[0]==1:
            grads=tmp_grad_weight.data.numpy().astype(np.float64)
            grad_aggregate_list.append(grads)
        return grad_aggregate_list

    @property
    def name(self):
        return 'fc_nn'