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

import master2 as master
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
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    # parser.add_argument('--diminishing-lr', type=ast.literal_eval, default=False, metavar='N',
    #                     help='set diminishing learning rate (default: False)')
    parser.add_argument('--diminishing-lr', type=int, default=0, 
                        help='set diminishing learning rate type (default: 0)\n\t0 <-- none\n\t1 <-- linear decrease (lr -= dimSize each step)\n\t2 <-- decrease exponentially with time (lr -= dimSize^step)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0)')
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
    parser.add_argument('--fault-thrshld', type=int, default=2, metavar='N',
                        help='how many number of worker nodes we want our filter to tolerate')
    parser.add_argument('--group-size', type=int, default=5, metavar='N',
                        help='in majority vote how many worker nodes are in a certain group')
    parser.add_argument('--compress-grad', type=str, default='compress', metavar='N',
                        help='compress/None indicate if we compress the gradient matrix before communication')
    parser.add_argument('--checkpoint-step', type=int, default=0, metavar='N',
                        help='which step to proceed the training process')
    parser.add_argument('--accumulative', type=ast.literal_eval, default=False, metavar='N',
                        help='to decide if use accumulative SGD')
    parser.add_argument('--accumulative-alpha', type=float, default=0, metavar='N',
                        help='accumulative SGD weight for historical gradients. If alpha=0, accumulate with equal weight for every gradient in history')
    parser.add_argument('--full-grad', type=ast.literal_eval, default=True, metavar='N',
                        help='to decide if the filter uses concatenated gradients (True) or natural pieces from networks (False)')
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
    parser.add_argument('--calculate-cosine', type=ast.literal_eval, default=False, metavar='N',
                        help='calculate or not the cosine distance between received gradients and the filtered gradient')
    parser.add_argument('--redundancy', action='store_true', default=False,
                        help='use redundancy filtering of byzantine workers, distribute identical batches')
    parser.add_argument('--p', type=float, default=.00,
                        help='the probability from 0 to 1 that byzantine workers will send errors in redundancy scheme')
    parser.add_argument('--q', type=float, default=1.00,
                        help='the probability from 0 to 1 that the master will check for errors in redundancy scheme')
    ''' add to distributed_nn.py '''
    parser.add_argument('--adapt-q', action='store_true',default=False,
                        help='Use adaptive fault-checking (increase over time)')
    parser.add_argument('--roll-freq',type=int,default=0,
                        help='frequency of storing rollback states; no rollback when 0')
    # parser.add_argument('',type=,default=,
    #                     help='')
    parser.add_argument('--targeted',action='store_true',default=False,
                        help='frequency of storing rollback states; no rollback when 0')
    parser.add_argument('--delay',type=int,default=0,
                        help='delay of Byzantine worker attacks (defaults=0)')

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


class CIFAR10SubLoader(datasets.CIFAR10):
    def __init__(self, *args, group_size=0, start_from=0, ovlp=False, **kwargs):
        super(CIFAR10SubLoader, self).__init__(*args, **kwargs)
        if group_size == 0:
            return
        if self.train:
            #print(self.train_data.shape)
            #print(self.train_labels.shape)
            if ovlp:
                all_labels = torch.tensor(self.targets)
                original_data = torch.tensor(self.data).clone()
                original_targets = all_labels.clone()
                for i in range(0,10):
                    i_idx = (all_labels == i)
                    if i==0:
                        self.data = torch.cat((original_data[i_idx][:int(start_from/10)], original_data[i_idx][int((start_from+group_size)/10):]))
                        self.targets = torch.cat((original_targets[i_idx][:int(start_from/10)], original_targets[i_idx][int((start_from+group_size)/10):]))
                    else:
                        self.data = torch.cat((self.data, original_data[i_idx][:int(start_from/10)], original_data[i_idx][int((start_from+group_size)/10):]))
                        self.targets = torch.cat((self.targets, original_targets[i_idx][:int(start_from/10)], original_targets[i_idx][int((start_from+group_size)/10):]))
            else:
                self.data = self.data[start_from:start_from + group_size]
                self.targets = self.targets[start_from:start_from + group_size]


def load_data(dataset, seed, args, rank, world_size):
    #print("here")
    torch.manual_seed(TORCH_SEED_)
    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
    #print("dataset: " + dataset)

    if dataset == "LinregTest":
        if rank==0:
            training_set = torch.load("linRegDataset")
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(190 / (world_size - 1))
                tmp_set = torch.load("linRegDataset")[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(TORCH_SEED_+rank)
                training_set = torch.load("linRegDataset")
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    else :
        if rank==0:
            training_set = torch.load(dataset)
            train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        else:
            if args.data_distribution == 'distributed':
                group_size = int(190 / (world_size - 1))
                tmp_set = torch.load(dataset)[group_size*(rank-1):group_size*rank]
                training_set = torch.utils.data.TensorDataset(tmp_set[0], tmp_set[1])
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
            elif args.data_distribution == 'same':
                torch.manual_seed(TORCH_SEED_+rank)
                training_set = torch.load(dataset)
                train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
        return train_loader, training_set, test_loader

    #print("here2")
    return None, None, None


def _generate_adversarial_nodes(args, world_size):
    np.random.seed(SEED_)
    if args.faulty_pattern == 'fixed':
        return [np.random.choice(np.arange(1, world_size), size=args.worker_fail, replace=False)] * (args.max_steps + 1)
    elif args.faulty_pattern == 'changing':
        return [np.random.choice(np.arange(1, world_size), size=args.worker_fail, replace=False) for _ in
                range(args.max_steps + 1)]
    elif args.faulty_pattern == 'median_of_means':
        b = math.ceil((world_size - 1) / (2*args.worker_fail+0.5))
        adversaries = [i*b for i in range(args.worker_fail)]
        assert len(adversaries) == args.worker_fail
        return [adversaries] * (args.max_steps + 1)


def prepare(args, rank, world_size):
    if args.mode=='multi_krum' and (args.multi_krum_m<=0 or args.multi_krum_m>world_size-args.worker_fail-1):
        raise Exception("multi-krum-m: Wrong number for multi-krum parameter m.")
    if args.fault_thrshld == None:
        raise Exception("fault-thrshld: Did not specify number of fault to be tolerated.")

    if args.approach == "baseline":
        # randomly select adversarial nodes
        adversaries = _generate_adversarial_nodes(args, world_size)
        print("Faulty agents:", adversaries[0], "Total:", len(adversaries[0]))
        train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=None, args=args, rank=rank,
                                                            world_size=world_size)
        
        # data_shape = training_set[0][0].size()[0]*training_set[0][0].size()[1]*training_set[0][0].size()[2]
        data_shape = training_set[0][0].size()[0]
        print("dataset size {}".format(data_shape))
        kwargs_master = {
            'redundancy': args.redundancy,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'diminishing_lr': args.diminishing_lr,
            'max_epochs': args.epochs,
            'max_steps': args.max_steps,
            'momentum': args.momentum,
            'network': args.network,
            'comm_method': args.comm_type,
            'worker_fail': args.worker_fail,
            'fault-thrshld': args.fault_thrshld,
            'eval_freq': args.eval_freq,
            'train_dir': args.train_dir,
            'update_mode': args.mode,
            'compress_grad': args.compress_grad,
            'checkpoint_step': args.checkpoint_step,
            'full_grad': args.full_grad,
            'total_size': data_shape,
            # 'channel': training_set[0][0].size()[0],
            # '1d_size': training_set[0][0].size()[1],
            'channel': 0,
            '1d_size': 0,
            'multi_krum_m': args.multi_krum_m,
            'grad_norm_keep_all': args.grad_norm_keep_all,
            'grad_norm_clip_n': args.grad_norm_clip_n,
            'calculate_cosine': args.calculate_cosine,
            'accumulative': args.accumulative,
            'accumulative_alpha': args.accumulative_alpha,
            # the following information is only used for simulating fault agents and not used by filters.
            'adversaries': adversaries,
            'err_mode': args.err_mode,
            'dataset_size': len(training_set),
            'q': args.q,
            'adapt_q': args.adapt_q,
            'roll_freq': args.roll_freq,
            'targeted': args.targeted
        }
        kwargs_worker = {
            'redundancy': args.redundancy,
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
            'total_size': data_shape,
            # 'channel': training_set[0][0].size()[0],
            # '1d_size': training_set[0][0].size()[1],
            'channel': 0,
            '1d_size': 0,
            'p': args.p,
            'delay': args.delay
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
        train_loader, training_set, test_loader = datum
        if rank == 0:
            master_fc_nn = master.SyncReplicaMaster_NN(comm=comm, **kwargs_master)
            master_fc_nn.build_model()
            print("Master node: the world size is {}, cur step: {}".format(master_fc_nn.world_size,
                                                                           master_fc_nn.cur_step))
            # _, val_split = torch.utils.data.random_split(training_set, [180, 10])
            # val_loader = torch.utils.data.DataLoader(val_split)
            # master_fc_nn.start(val_loader)
            master_fc_nn.start(train_loader)
            # master_fc_nn.start()
            print("Done sending massage to workers!")
        else:
            worker_fc_nn = worker.DistributedWorker(comm=comm, **kwargs_worker)
            worker_fc_nn.build_model()
            print("Worker node: {} in all {}, next step: {}".format(worker_fc_nn.rank, worker_fc_nn.world_size,
                                                                    worker_fc_nn.next_step))
            #for indx, (img, target) in enumerate(train_loader):
            #    if indx < 5:
            #        print(worker_fc_nn.rank,indx,target)

            worker_fc_nn.train(train_loader=train_loader, test_loader=test_loader)

            print("Worrker {} Now the next step is: {}".format(rank,worker_fc_nn.next_step))
            
