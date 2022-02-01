import ast
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
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=10000, metavar='N',
                        help='the maximum number of iterations')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--diminishing-lr', type=ast.literal_eval, default=False, metavar='N',
                        help='set diminishing learning rate (default: False)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
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
    parser.add_argument('--async-thrshld', type=int, default=2, metavar='N',
                        help='how many stragglers we intend to tolerate')
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
    parser.add_argument('--multi-krum-m', type=int, default=1, metavar='N',
                        help='parameter m in multi-krum. Positive, default 1, no large than n-2f-1')
    parser.add_argument('--grad-norm-keep-all', type=ast.literal_eval, default=False, metavar='N',
                        help='decide if when using gradient norm clipping, keep all gradients (True) or throw away the largest ones (False)')
    parser.add_argument('--grad-norm-clip-n', type=int, default=1, metavar='N',
                        help='specifying parameter n when using gradient norm clipping (multi-parts) with n piece')
    parser.add_argument('--save-honest-list', type=ast.literal_eval, default=False, metavar='N',
                        help='decide whether or not saving the honest agent list')
    parser.add_argument('--omit-agents', type=ast.literal_eval, default=False, metavar='N',
                        help='decide whether to remove data according to corresponding agents')
    parser.add_argument('--faulty-list', nargs='*', type=int, metavar='N',
                        help='input to specify faulty agent list (range: 1 to number of agents)')
    parser.add_argument('--zero-initial-weights', type=ast.literal_eval, default=True, metavar='N',
                        help='decide if to use 0 vector as inital weights')
    args = parser.parse_args()
    return args


class MNISTSubLoader(datasets.MNIST):
    def __init__(self, *args, group_size=0, start_from=0, **kwargs):
        super(MNISTSubLoader, self).__init__(*args, **kwargs)
        if group_size == 0:
            return
        if self.train:
            #print(self.train_data.shape)
            #print(self.train_labels.shape)
            self.data = self.data[start_from:start_from + group_size]
            self.targets = self.targets[start_from:start_from + group_size]


def load_data(dataset, seed, args, rank, world_size):
    print("here")
    torch.manual_seed(TORCH_SEED_)
    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
    else:
        seed = TORCH_SEED_
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
                torch.manual_seed(seed+rank)
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

    elif dataset == "LinReg":
        if rank==0:
            training_set = torch.load("linRegDataset")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(10000 / (world_size - 1))
                tmp_set = torch.load("linRegDataset")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("linRegDataset")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "LinReg2":
        if rank==0:
            training_set = torch.load("linRegDataset2")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(190 / (world_size - 1))
                tmp_set = torch.load("linRegDataset2")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("linRegDataset2")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "LinReg3":
        if rank==0:
            training_set = torch.load("linRegDataset3")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(190 / (world_size - 1))
                tmp_set = torch.load("linRegDataset3")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("linRegDataset3")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "LinReg4":
        if rank==0:
            training_set = torch.load("linRegDataset4")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(1900 / (world_size - 1))
                tmp_set = torch.load("linRegDataset4")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("linRegDataset4")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "LinReg5":
        if rank==0:
            training_set = torch.load("linRegDataset5")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(1900 / (world_size - 1))
                tmp_set = torch.load("linRegDataset5")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("linRegDataset5")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "ApproxReg1":
        if rank==0:
            training_set = torch.load("approximationDataset1")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(6 / (world_size - 1))
                tmp_set = torch.load("approximationDataset1")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("approximationDataset1")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "ApproxReg2":
        if rank==0:
            training_set = torch.load("approximationDataset2")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(6 / (world_size - 1))
                tmp_set = torch.load("approximationDataset2")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("approximationDataset2")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "ApproxReg3":
        if rank==0:
            training_set = torch.load("approximationDataset3")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(100 / (world_size - 1))
                tmp_set = torch.load("approximationDataset3")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("approximationDataset3")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "ApproxReg4":
        if rank==0:
            training_set = torch.load("approximationDataset4")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(6 / (world_size - 1))
                tmp_set = torch.load("approximationDataset4")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("approximationDataset4")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "ApproxReg5":
        if rank==0:
            training_set = torch.load("approximationDataset5")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(6 / (world_size - 1))
                tmp_set = torch.load("approximationDataset5")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("approximationDataset5")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "ApproxReg6":
        if rank==0:
            training_set = torch.load("approximationDataset6")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(10 / (world_size - 1))
                tmp_set = torch.load("approximationDataset6")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("approximationDataset6")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "PreciseReg":
        if rank==0:
            training_set = torch.load("regressionDataset")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(6 / (world_size - 1))
                tmp_set = torch.load("regressionDataset")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("regressionDataset")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "SVMData":
        if rank==0:
            training_set = torch.load("svmDataset")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(1000 / (world_size - 1))
                tmp_set = torch.load("svmDataset")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("svmDataset")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    elif dataset == "WDBC":
        if rank==0:
            training_set = torch.load("wdbcDataset")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(400 / (world_size - 1))
                tmp_set = torch.load("wdbcDataset")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(seed+rank)
                training_set = torch.load("wdbcDataset")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader
    #print("here2")
    return None, None, None


def _generate_adversarial_nodes(args, world_size):
    if args.seed is None:
        np.random.seed(SEED_)
    else:
        np.random.seed(args.seed)
    if args.faulty_pattern == 'changing' or 'async' in args.err_mode:
        return [np.random.choice(np.arange(1, world_size), size=args.worker_fail, replace=False) for _ in
                range(args.max_steps + 1)]
    elif args.faulty_pattern == 'fixed':
        if args.faulty_list is None:
            return [np.random.choice(np.arange(1, world_size), size=args.worker_fail, replace=False)] * (args.max_steps + 1)
        else:
            if len(args.faulty_list)==args.worker_fail and max(args.faulty_list)<world_size and min(args.faulty_list)>0:
                return [np.array(args.faulty_list)] * (args.max_steps + 1)
            else:
                raise Exception("Wrong list of faulty agents")
    elif args.faulty_pattern == 'median_of_means':
        b = math.ceil((world_size - 1) / (2*args.worker_fail+0.5))
        adversaries = [i*b for i in range(args.worker_fail)]
        print(b, adversaries)
        assert len(adversaries) == args.worker_fail
        return [adversaries] * (args.max_steps + 1)


def prepare(args, rank, world_size):
    if args.mode=='multi_krum' and (args.multi_krum_m<=0 or args.multi_krum_m>world_size-args.worker_fail-1):
        raise Exception("Wrong number for multi-krum parameter m.")

    if args.approach == "baseline":
        # randomly select adversarial nodes
        adversaries = _generate_adversarial_nodes(args, world_size)
        if args.save_honest_list and rank == 0:
            np.save(args.train_dir+"honest_list", np.delete(np.arange(world_size-1), np.array(adversaries[0]-1)))
        print("Faulty agents:", adversaries[0], "Total:", len(adversaries[0]))
        train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=args.seed, args=args, rank=rank,
                                                            world_size=world_size)
        data_shape = training_set[0][0].size()[0]
        print("datashape=",training_set[0][0].size())
        kwargs_master = {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'diminishing_lr': args.diminishing_lr,
            'max_epochs': args.epochs,
            'max_steps': args.max_steps,
            'momentum': args.momentum,
            'network': args.network,
            'comm_method': args.comm_type,
            'worker_fail': args.worker_fail,
            'async_thrshld': args.async_thrshld,
            'eval_freq': args.eval_freq,
            'train_dir': args.train_dir,
            'update_mode': args.mode,
            'compress_grad': args.compress_grad,
            'checkpoint_step': args.checkpoint_step,
            'data_size': data_shape,
            'multi_krum_m': args.multi_krum_m,
            'grad_norm_keep_all': args.grad_norm_keep_all,
            'grad_norm_clip_n': args.grad_norm_clip_n,
            'zero_initial_weights': args.zero_initial_weights,
            # the following information is only used for simulating fault agents and not used by filters.
            'adversaries': adversaries,
            'err_mode': args.err_mode,
            'omit_agents': args.omit_agents
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
    # print(train_loader, training_set, test_loader)
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
