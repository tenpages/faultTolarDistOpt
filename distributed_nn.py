import sys
import math
import threading
import argparse
import time
import random

from mpi4py import MPI

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

import numpy as np

from torchvision import datasets, transforms

import master
import worker

SEED_ = 428
TORCH_SEED_ = 761


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=10000, metavar='N',
                        help='the maximum number of iterations')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--network', type=str, default='LeNet', metavar='N',
                        help='which kind of network we are going to use, support LeNet and ResNet currently')
    parser.add_argument('--mode', type=str, default='normal', metavar='N',
                        help='determine if we use normal averaged gradients or geometric median (in normal mode)\
                         or whether we use normal/majority vote in coded mode to udpate the model')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    parser.add_argument('--comm-type', type=str, default='Bcast', metavar='N',
                        help='which kind of method we use during the mode fetching stage')
    parser.add_argument('--err-mode', type=str, default='rev_grad', metavar='N',
                        help='which type of byzantine err we are going to simulate rev_grad/constant/random are supported')
    parser.add_argument('--approach', type=str, default='maj_vote', metavar='N',
                        help='method used to achieve byzantine tolerance, currently majority vote is supported\
                         set to normal will return to normal mode')
    parser.add_argument('--num-aggregate', type=int, default=5, metavar='N',
                        help='how many number of gradients we wish to gather at each iteration')
    parser.add_argument('--eval-freq', type=int, default=50, metavar='N',
                        help='it determines per how many step the model should be evaluated')
    parser.add_argument('--train-dir', type=str, default='output/models/', metavar='N',
                        help='directory to save the temp model during the training process for evaluation')
    parser.add_argument('--adversarial', type=int, default=1, metavar='N',
                        help='how much adversary we want to add to a certain worker')
    parser.add_argument('--worker-fail', type=int, default=2, metavar='N',
                        help='how many number of worker nodes we want to simulate byzantine error on')
    parser.add_argument('--group-size', type=int, default=5, metavar='N',
                        help='in majority vote how many worker nodes are in a certain group')
    parser.add_argument('--compress-grad', type=str, default='compress', metavar='N',
                        help='compress/None indicate if we compress the gradient matrix before communication')
    parser.add_argument('--checkpoint-step', type=int, default=0, metavar='N',
                        help='which step to proceed the training process')
    parser.add_argument('--faulty-pattern', type=str, default='fixed', metavar='N',
                        help='decide faulty gradients are send from "fixed" workers or "changing" workers each step')
    parser.add_argument('--data-distribution', type=str, default='same', metavar='N',
                        help='decide if data is "distributed" among workers or every worker owns the "same" data')
    args = parser.parse_args()
    return args


class MNISTSubLoader(datasets.MNIST):
    def __init__(self, *args, group_size=0, start_from=0, **kwargs):
        super(MNISTSubLoader, self).__init__(*args, **kwargs)
        if group_size == 0:
            return
        if self.train:
            print(self.train_data.shape)
            print(self.train_labels.shape)
            self.data = self.data[start_from:start_from + group_size]
            self.targets = self.targets[start_from:start_from + group_size]


def load_data(dataset, seed, args, rank, world_size):
    print("here")
    torch.manual_seed(TORCH_SEED_)
    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
    print("dataset: " + dataset)
    if dataset == "MNIST":
        if rank==0:
            training_set = datasets.MNIST('./mnist_data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(60000 / (world_size - 1))
                training_set = MNISTSubLoader('./mnist_data_sub/'+str(rank), train=True, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]), group_size=group_size, start_from=group_size*(rank-1))
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                training_set = datasets.MNIST('./mnist_data', train=True, download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "CIFAR10":
        training_set = datasets.CIFAR10('./cifar10_data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None

        return train_loader, training_set, test_loader

    print("here2")
    return None, None, None


def _generate_adversarial_nodes(args, world_size):
    np.random.seed(SEED_)
    if args.faulty_pattern == 'fixed':
        return [np.random.choice(np.arange(1, world_size), size=args.worker_fail, replace=False)] * (args.max_steps + 1)
    elif args.faulty_pattern == 'changing':
        return [np.random.choice(np.arange(1, world_size), size=args.worker_fail, replace=False) for _ in
                range(args.max_steps + 1)]


def prepare(args, rank, world_size):
    if args.approach == "baseline":
        # randomly select adversarial nodes
        adversaries = _generate_adversarial_nodes(args, world_size)
        train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=None, args=args, rank=rank,
                                                            world_size=world_size)
        data_shape = training_set[0][0].size()[0]*training_set[0][0].size()[1]*training_set[0][0].size()[2]
        kwargs_master = {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'max_epochs': args.epochs,
            'max_steps': args.max_steps,
            'momentum': args.momentum,
            'network': args.network,
            'comm_method': args.comm_type,
            'worker_fail': args.worker_fail,
            'eval_freq': args.eval_freq,
            'train_dir': args.train_dir,
            'update_mode': args.mode,
            'compress_grad': args.compress_grad,
            'checkpoint_step': args.checkpoint_step,
            'data_size': data_shape
        }
        kwargs_worker = {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'max_epochs': args.epochs,
            'max_steps': args.max_steps,
            'momentum': args.momentum,
            'network': args.network,
            'comm_method': args.comm_type,
            'adversary': args.adversarial,
            'worker_fail': args.worker_fail,
            'err_mode': args.err_mode,
            'compress_grad': args.compress_grad,
            'eval_freq': args.eval_freq,
            'train_dir': args.train_dir,
            'checkpoint_step': args.checkpoint_step,
            'adversaries': adversaries,
            'data_size': data_shape
        }
    print(train_loader, training_set, test_loader)
    datum = (train_loader, training_set, test_loader)
    return datum, kwargs_master, kwargs_worker


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args = add_fit_args(argparse.ArgumentParser(description="Draco"))

    datum, kwargs_master, kwargs_worker = prepare(args, rank, world_size)
    if args.approach == 'baseline':
        train_loader, _, test_loader = datum
        if rank == 0:
            master_fc_nn = master.SyncReplicaMaster_NN(comm=comm, **kwargs_master)
            master_fc_nn.build_model()
            print("Master node: the world size is {}, cur step: {}".format(master_fc_nn.world_size,
                                                                           master_fc_nn.cur_step))
            master_fc_nn.start()
            print("Done sending massage to workers!")
        else:
            worker_fc_nn = worker.DistributedWorker(comm=comm, **kwargs_worker)
            worker_fc_nn.build_model()
            print("Worker node: {} in all {}, next step: {}".format(worker_fc_nn.rank, worker_fc_nn.world_size,
                                                                    worker_fc_nn.next_step))
            worker_fc_nn.train(train_loader=train_loader, test_loader=test_loader)
            print("Now the next step is: {}".format(worker_fc_nn.next_step))
