from __future__ import print_function
import os.path
import time
import argparse
from datetime import datetime
import copy

from mpi4py import MPI
import numpy as np

from model_ops.fc import Full_Connected
from model_ops.lenet import LeNet
from model_ops.resnet import ResNet18
from model_ops.vgg import VGG19
from nn_ops import NN_Trainer

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Validation settings
    parser.add_argument('--eval-batch-size', type=int, default=10000, metavar='N',
                        help='the batch size when doing model validation, complete at once on default')
    parser.add_argument('--eval-freq', type=int, default=50, metavar='N',
                        help='it determines per how many step the model should be evaluated')
    parser.add_argument('--model-dir', type=str, default='output/models/', metavar='N',
                        help='directory to save the temp model during the training process for evaluation')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    parser.add_argument('--network', type=str, default='LeNet', metavar='N',
                        help='which kind of network we are going to use, support LeNet and ResNet currently')
    args = parser.parse_args()
    return args


class DistributedEvaluator(object):
    '''
    The DistributedEvaluator aims at providing a seperate node in the distributed cluster to evaluate
    the model on validation/test set and return the results
    In this version, the DistributedEvaluator will only load the model from the dir where the master
    save the model and do the evaluation task based on a user defined frequency
    '''

    def __init__(self, **kwargs):
        self._cur_step = 0
        self._model_dir = kwargs['model_dir']
        self._eval_freq = int(kwargs['eval_freq'])
        self._eval_batch_size = kwargs['eval_batch_size']
        self.network_config = kwargs['network']
        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []
        if self.network_config == "FC":
            self.network = Full_Connected(kwargs['input_size'])
        elif self.network_config == 'LeNet':
            self.network = LeNet(kwargs['channel'], kwargs['1d_size'])
        elif self.network_config == 'ResNet18':
            self.network = ResNet18(kwargs['channel'])
        elif self.network_config == 'VGG19':
            self.network = VGG19(kwargs['channel'])

    def evaluate(self, validation_loader):
        # init objective to fetch at the begining
        self._next_step_to_fetch = self._cur_step + self._eval_freq
        self._num_batch_per_epoch = len(validation_loader) / self._eval_batch_size
        # check if next temp model exsits, if not we wait here else
        # we continue to do the model evaluation
        while True:
            model_dir_ = self._model_dir_generator(self._next_step_to_fetch)
            print("loading model from directory ".format(model_dir_))
            if os.path.isfile(model_dir_):
                self._load_model(model_dir_)
                print("Evaluator evaluating results on step {}".format(self._next_step_to_fetch))
                self._evaluate_model(validation_loader)
                self._next_step_to_fetch += self._eval_freq
            else:
                break
                # TODO(hwang): sleep appropriate period of time make sure to tune this parameter
                # time.sleep(10)
        print("finished evaluation.")

    def _evaluate_model(self, test_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        prec1_counter_ = prec3_counter_ = batch_counter_ = 0
        for data, y_batch in test_loader:
            data, target = Variable(data, volatile=True), Variable(y_batch)
            output = self.network(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            prec1_tmp, prec3_tmp = accuracy(output.data, y_batch, topk=(1, 3))
            prec1_counter_ += prec1_tmp.numpy()[0]
            prec3_counter_ += prec3_tmp.numpy()[0]
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec3 = prec3_counter_ / batch_counter_
        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.8f}, Prec@1: {} Prec@3: {}'.format(test_loss, prec1, prec3))

    def _load_model(self, file_path):
        with open(file_path, "rb") as f_:
            self.network.load_state_dict(torch.load(f_))

    def _model_dir_generator(self, next_step_to_fetch):
        return self._model_dir + "model_step_" + str(next_step_to_fetch)


if __name__ == "__main__":
    # this is only a simple test case
    args = add_fit_args(argparse.ArgumentParser(description='PyTorch Distributed Evaluator'))

    # load training and test set here:
    if args.dataset == "MNIST":
        testing_set=datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_loader = torch.utils.data.DataLoader(testing_set, batch_size=args.eval_batch_size, shuffle=True)
        data_shape = testing_set[0][0].size()[0]*testing_set[0][0].size()[1]*testing_set[0][0].size()[2]
    elif args.dataset == "CIFAR10":
        testing_set = datasets.CIFAR10('./cifar10_data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        test_loader = torch.utils.data.DataLoader(testing_set, batch_size=args.eval_batch_size, shuffle=True)
        data_shape = testing_set[0][0].size()[0]*testing_set[0][0].size()[1]*testing_set[0][0].size()[2]
    print("testing set loaded.")

    kwargs_evaluator = {'model_dir': args.model_dir, 'eval_freq': args.eval_freq,
                        'eval_batch_size': args.eval_batch_size, 'network': args.network,
                        'input_size': data_shape, 'channel': testing_set[0][0].size()[0],
                        '1d_size': testing_set[0][0].size()[1]}
    evaluator_nn = DistributedEvaluator(**kwargs_evaluator)
    print("evaluator initiated.")
    evaluator_nn.evaluate(validation_loader=test_loader)
