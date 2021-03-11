import time
import sys, os
from sys import getsizeof
import numpy as np
import hdmedians as hd
import torch
from torch import nn
from torch.autograd import Variable
import random

import math
from math import ceil as ceil

from mpi4py import MPI

from scipy.spatial.distance import cosine
import csv

from scipy.special import erfinv
from numpy.random import randn
from math import sqrt

from compress_gradient import decompress
from model_ops.lenet import LeNet_Split
from model_ops.fc import Full_Connected_Split
from model_ops.resnet import ResNet18
from model_ops.resnetn import ResNet18N
from model_ops.vgg import VGG13, VGG16, VGG19
from nn_ops import NN_Trainer, accuracy
from optim.sgd_modified import SGDModified

from functools import reduce

from model_ops.reg_fc import LinregTest_Split 

import math

STEP_START_ = 1


class SyncReplicaMaster_NN(NN_Trainer):
    """
    Register master node using this class
    """

    def __init__(self, comm, **kwargs):
        self.comm = comm  # get MPI communicator object
        self.world_size = comm.Get_size()
        self.num_workers = self.world_size - 1
        self.coded_buffer = []
        self.cur_step = STEP_START_
        self.lr = kwargs['learning_rate']
        self._diminishing_lr = kwargs['diminishing_lr']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']

        self._redundancy = kwargs['redundancy']
        self._q = kwargs['q']
        ''' temporary '''
        self._roll_freq = kwargs['roll_freq'] 
        self._rollback = {'loss': None, 'params': None}
        self._adaptive = kwargs['adapt_q']
        self._targeted = kwargs['targeted']

        self._byzantine_workers = []
        self._num_grad_to_collect = self.world_size - 1
        self._grad_aggregate_buffer = []
        self._historical_buffer = []
        self._model_shapes = []
        self._first_grad_received = False
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._max_steps = kwargs['max_steps']
        self._update_mode = kwargs['update_mode']
        self._compress_grad = kwargs['compress_grad']
        self._checkpoint_step = kwargs['checkpoint_step']
        self._s = kwargs['worker_fail']
        self._t = kwargs['fault-thrshld']
        self._full_grad = kwargs['full_grad']
        self._total_size = kwargs['total_size']
        self._channel = kwargs['channel']
        self._size = kwargs['1d_size']
        self._multi_krum_m = kwargs['multi_krum_m']
        self._grad_norm_keep_all = kwargs['grad_norm_keep_all']
        self._grad_norm_clip_n = kwargs['grad_norm_clip_n']
        self._calculate_cosine = kwargs['calculate_cosine']

        self._accumulative = kwargs['accumulative']
        self._accumulative_alpha = kwargs['accumulative_alpha']

        # the following information is only used for simulating fault agents and not used by filters.
        self._adversaries = kwargs['adversaries']
        self._err_mode = kwargs['err_mode']
        self.dataset_size = kwargs['dataset_size']
        self.batch_size = kwargs['batch_size']

        # self.validation_criterion = nn.CrossEntropyLoss()
        if self.network_config == 'LinregTest':
            self.validation_criterion = nn.MSELoss()
        else:
            self.validation_criterion = nn.CrossEntropyLoss()

    def build_model(self) :
        # print("building model, self._size ", self._size)
        if self.network_config == "FC":
            self.network = Full_Connected_Split(self._total_size)
        elif self.network_config == "LeNet":
            self.network = LeNet_Split(self._channel,self._size)
        elif self.network_config == "ResNet18":
            self.network = ResNet18(self._channel)
        elif self.network_config == "ResNet18N":
            self.network = ResNet18N(self._channel)
        elif self.network_config == 'VGG13':
            self.network = VGG13(self._channel)
        elif self.network_config == 'VGG16':
            self.network = VGG16(self._channel)
        elif self.network_config == 'VGG19':
            self.network = VGG19(self._channel)
        elif self.network_config == 'LinregTest':
            self.network = LinregTest_Split(self._total_size)

        if self._checkpoint_step != 0:
            file_path = self._train_dir + "model_step_" + str(self._checkpoint_step)
            self._load_model(file_path)
            self.cur_step = int(self._checkpoint_step) + 1

        # gradient accumulator collects gradients from worker nodes
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size - 1, mode=self._compress_grad)
        self.init_model_shapes()
        # optimizer can be others
        self.optimizer = SGDModified(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        # if self._diminishing_lr == True:
        if self._diminishing_lr:
            # lr_lambda = lambda epoch: 10/((epoch/1000)+1)
            ''' diminishing learning rate-- half lr after epoch 200 '''
            if self._diminishing_lr == 1:
                dimin_size = self.lr/(self._max_steps + 1)
                lr_lambda = lambda epoch: (1.0 - (dimin_size*epoch)/self.lr)
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lr_lambda)
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=.5)
            elif self._diminishing_lr == 2:
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=.8)
            else:
                print('ERROR: invalid --diminish-lr input')

    def start(self, validation_loader):
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})

        if os.path.exists(self._train_dir+"logfile.txt"):
            print("Master... removing {}logfile.txt".format(self._train_dir))
            os.remove(self._train_dir+"logfile.txt")

        self.async_bcast_step()
        if self._redundancy :
            # self.bcast_rand_nums()
            self.bcast_seed()

        if self._checkpoint_step != 0:
            # torch.set_rng_state(torch.load(self._train_dir+"rng_state_"+str(self._checkpoint_step)))
            self.optimizer.load_state_dict(torch.load(self._train_dir+"optim_"+str(self._checkpoint_step)))
            # if self._diminishing_lr == True:
            if self._diminishing_lr:
                self.scheduler.load_state_dict(torch.load(self._train_dir+"scheduler_"+str(self._checkpoint_step)))

        ''' redundancy metrics '''
        total_grads = 0
        used_grads = 0
        comp_eff = [[],[],[]]
        total_rolls = 0

        prev_params = None
        prev_grad_avg = None
        param_diffs = []
        grad_diffs = []

        for i in range(self._checkpoint_step + 1, self._max_steps + 1):
            # if self._diminishing_lr == True:
            if self._diminishing_lr:
                self.scheduler.step()
                print("[DEBUG] Master learning rate: ",self.optimizer.param_groups[0]['lr'])
            self.network.train()
            self.optimizer.zero_grad()
            self._first_grad_received = False
            enough_gradients_received = False

            assert (i == self.cur_step)
            print("Master node is entering step: {}".format(i))

            self.async_bcast_step()
            total_grads_t = 0
            used_grads_t = 0
            fault_check = None

            if self._redundancy:
                # q_decision 
                fault_check = self.q_decision(self._q)
                dp_list, worker_list = (None,None)
                # print("_q == {} {}".format(self._q,type(self._q)))
                if self._q == 1.0 or fault_check:
                    fault_check == 1 
                    self._num_grad_t_collect = (self.batch_size)*(self._s + 1)
                    dp_list, worker_list = self.async_bcast_datapoints_redundant()
                else:
                    self._num_grad_to_collect = self.batch_size
                    dp_list, worker_list = self.async_bcast_datapoints_singular()

                self.set_coded_buffer(worker_list)
            
            if self.comm_type == 'Bcast':
                self.async_bcast_layer_weights_bcast()
            elif self.comm_type == 'Async':
                self.async_bcast_layer_weights_async()

            if self._redundancy:
                print(f"Master {self.cur_step} Byzantine Workers: {self._byzantine_workers}")

                gradient_fetch_requests = self.fetch_coded_gradients_start(worker_list)
                statuses = [MPI.Status() for _ in gradient_fetch_requests]
                
                MPI.Request.Waitall(requests=gradient_fetch_requests, statuses=statuses)
                faulty_grad_datapoints=[]
                faulty_worker_ranks = []

                # ''' DEBUG print grads last 5 steps '''
                # print("[DEBUG] Master [{}] grads".format(self.cur_step))
                # if self._max_steps - self.cur_step < 5:
                #     for dpidx, dp_buffer in enumerate(self.coded_buffer):
                #         print(dp_buffer[0])

                if self._targeted:
                    if prev_grad_avg is None:
                        print("first grad avg")
                        # prev_grad_avg = np.mean(self.coded_buffer,2)
                        # for worker in self.coded_buffer[0]:
                        #     for layer in worker:
                        #         print("layer size", len(layer))
                        prev_grad_avg = np.array([[[[np.mean(layer)] for layer in worker] for worker in dp] for dp in self.coded_buffer])
                        print(prev_grad_avg.shape)
                    else:
                        # print("gradient avg change:", np.mean(self.coded_buffer,2)-prev_grad_avg)
                        # prev_grad_avg = np.mean(self.coded_buffer,2)
                        # print("gradient avg change:", np.mean(self.coded_buffer,1)-prev_grad_avg)
                        # grad_diffs.append(np.abs(prev_grad_avg - np.mean(np.mean(self.coded_buffer))))
                        grad_diffs.append(np.mean(np.abs(prev_grad_avg - np.array([[[[np.mean(layer)] for layer in worker] for worker in dp] for dp in self.coded_buffer]))))
                        # prev_grad_avg = np.mean(self.coded_buffer,1)
                        # prev_grad_avg = np.mean(np.mean(self.coded_buffer))
                        prev_grad_avg = np.array([[[[np.mean(layer)] for layer in worker] for worker in dp] for dp in self.coded_buffer])

                if self._s > 0 and (self._q == 1.0 or fault_check):
                    print("Master step {} check for faults".format(self.cur_step))
                    for dpidx, dp_buffer in enumerate(self.coded_buffer):
                        dp_flag = True
                        first=dp_buffer[0]
                        # print("dp_buffer[0]",dp_buffer[0])

                        for widx, worker_buffer in enumerate(dp_buffer):
                            for layer_idx, param in enumerate(worker_buffer):
                                # if gradients of datapoint dpidx do not match within a tolerance,
                                # then add to list of faulty_grad_datapoints
                                if not np.allclose(worker_buffer[layer_idx],first[layer_idx], rtol=1e-02,atol=1e-02) :
                                   faulty_grad_datapoints.append(dpidx)
                                   dp_flag=False
                                   break
                            if not dp_flag:
                                break
                else:
                    print(f"Master step {self.cur_step} skipping fault check")
                    ''' DEBUG
                        targeted fault-checking
                        using gradient norm
                    '''
                    if self._targeted:
                        mu = np.mean(grad_diffs[:len(grad_diffs)-1])
                        std = np.std(grad_diffs[:len(grad_diffs)-1])
                        # if self.cur_step > 20 and abs(grad_diffs[-1] - mu)/std >= 5.0:
                            # print('[DEBUG] {}\ttargeted fault-check'.format(self.cur_step))

                        if self._targeted and self.cur_step > 20:
                            print('[DEBUG]',self.cur_step,abs(mu-grad_diffs[-1])/std)

                if faulty_grad_datapoints:
                    print(f"Master step {self.cur_step} fault caught\ndatapoints with faulty gradients: {faulty_grad_datapoints}") 
                    # self._num_grad_to_collect = (self._s+1)*self.batch_size + self._s
                    # redistribute datapoints for second round 
                    # rule: datapoints must go to different workers than before
                    new_dp_list, new_worker_list = self.async_bcast_redundant_datapoints(dp_list, faulty_grad_datapoints)
                    print('old_worker_list', worker_list)
                    print("new_worker_list",new_worker_list)

                    combined_list = [[] for _ in new_worker_list]
                    for dp in range(len(worker_list)):
                        for rank in worker_list[dp]:
                            combined_list[dp].append(rank)
                        for rank in new_worker_list[dp]:
                            combined_list[dp].append(rank)
                    print("updated worker rank list: {}".format(combined_list))

                    # create requests for redundant gradients and wait for messages
                    redundant_fetch_requests = self.fetch_coded_gradients_redundant(new_worker_list, worker_list)
                    statuses = [MPI.Status() for _ in redundant_fetch_requests]
                    MPI.Request.Waitall(requests=redundant_fetch_requests, statuses=statuses)
                    
                    # determine the correct grad via majority vote
                    for dpidx in range(self.batch_size):
                        curr_faulty_workers = []
    
                        # create dictionary grouping worker indexes for datapoint dpidx
                        # by gradients sent from that worker
                        if dpidx in faulty_grad_datapoints:
                            grad_dict = {}  # key = widx (index of some worker in self.coded_buffer[dpidx]
                                            # value = list of ranks with the key-th worker's gradient
                            for widx, grads in enumerate(self.coded_buffer[dpidx]):
                                if  len(grad_dict) == 0 :
                                    grad_dict[widx] = [widx]
                                else :
                                    grad_match = False
                                    for key in grad_dict:
                                        params_match = True
                                        for pidx, param in enumerate(self.network.parameters()) :
                                            """
                                            requires more absolute tolerance (atol=...)
                                            as a result of rounding in different
                                            networks-- is this an issue?
                                            """
                                            if not np.allclose(self.coded_buffer[dpidx][widx][pidx], self.coded_buffer[dpidx][key][pidx],rtol=1e-02,atol=1e-03) :
                                                params_match = False
                                                break
                                        if params_match :
                                            grad_dict[key].append(widx)
                                            grad_match = True
                                            break
                                    if not grad_match :
                                        grad_dict[widx] = [widx]
                                        
                            # grad_dict maps indexes of some workers to list of all workers with same gradient (including itself)
                            max_size=-1
                            max_size_key=-1
                            for key in grad_dict :
                                # find correct gradients by majority vote
                                if len(grad_dict[key]) > max_size :
                                    max_size = len(grad_dict[key])
                                    max_size_key = key
                            # print(f"Datapoint [{dpidx}] {max_size}/{len(self.coded_buffer[dpidx])} workers agree")
                            for key in grad_dict :
                                if key != max_size_key :
                                    for widx in grad_dict[key] :
                                        curr_faulty_workers.append(combined_list[dpidx][widx])
                                        rank = combined_list[dpidx][widx]

                                        # add rank of faulty worker to faulty_worker_ranks
                                        if rank not in faulty_worker_ranks:
                                            faulty_worker_ranks.append(rank)

                            # print(f"Master step {self.cur_step} faulty workers from dp {dpidx} : {curr_faulty_workers}")
                            
                    print(f"Master: step {self.cur_step} faulty worker ranks: ",faulty_worker_ranks)

                    # aggregate gradients of all non-faulty workers
                    for dpidx, grad_list in enumerate(self.coded_buffer):
                        used_grads_t = used_grads_t + 1
                        for widx, grad in enumerate(grad_list):
                            total_grads_t = total_grads_t + 1
                            if combined_list[dpidx][widx] not in faulty_worker_ranks:
                                for layer_index, param in enumerate(self.network.parameters()):
                                    self.aggregate_gradient(gradient=grad[layer_index], layer_idx=layer_index, source=combined_list[dpidx][widx] - 1)

                    # add ranks of Byzantine workers to blacklist and reduce f = "possible Byzantine workers"
                    for rank in faulty_worker_ranks:
                        self._byzantine_workers.append(rank)
                        self._s -= 1
                    
                   
                    # fault was caught-- load rollback state if preferred
                    val_loss = self._evaluate_model(validation_loader)
                    if self._roll_freq and self._rollback['loss']:
                        if val_loss > self._rollback['loss']:
                            val_loss = self._evaluate_model(validation_loader)
                            self._load_rollback()
                            total_rolls += 1
                            print("[DEBUG] loss at time of fault", val_loss)
                            new_loss =  self._evaluate_model(validation_loader)
                            print("[DEBUG] new loss",new_loss)
                    else:
                        print("Master {} no rollback to load".format(self.cur_step))
                            
                    print(f"Master {self.cur_step} used gradients: {used_grads_t}/{total_grads_t}")

                else :
                    # either skipped fault check or found no faulty gradients
                    # send blank lists to skip redundant step
                    redundancy_requests=[]
                    for rank in range(1,self.world_size):
                        # send an empty list to each worker with tag=9
                        # if rank not in self._byzantine_workers:
                        redundancy_requests.append(self.comm.isend([], dest=rank, tag=8))
                    for i in range(len(redundancy_requests)):
                        redundancy_requests[i].wait()
                    count = 0
                    # aggregate gradient
                    for dpidx, grad_list in enumerate(self.coded_buffer):
                        # tally total gradients used for update
                        # for each dpidx, accumulated grads are averaged
                        used_grads_t = used_grads_t + 1

                        if len(grad_list) == 1:
                            total_grads_t = total_grads_t + 1
                            grad = grad_list[0]
                            count += 1
                            for layer_index, param in enumerate(self.network.parameters()):
                                #if layer_index < 2:
                                #    print("[{}] worker={} layer={}\t{}".format(self.cur_step,worker_list[dpidx][0],layer_index,grad[layer_index].tolist()))
                                self.aggregate_gradient(gradient=grad[layer_index], layer_idx=layer_index, source=worker_list[dpidx][0] - 1)
                        else:
                            if dpidx == 0:
                                print(f"Master {self.cur_step} no faulty gradients")

                            for widx, grad in enumerate(grad_list):
                                # add to total gradients calculated, greater than or equal to (batch_size) * (f+1)
                                total_grads_t = total_grads_t + 1
                                for layer_index, param in enumerate(self.network.parameters()):
                                    self.aggregate_gradient(gradient=grad[layer_index], layer_idx=layer_index, source=worker_list[dpidx][widx] - 1)

                    print(f"Master {self.cur_step} used gradients: {used_grads_t}/{total_grads_t}")
                
                # log computational efficieny = (used grads) / (calculated grads)
                comp_eff[1].append(used_grads_t)
                comp_eff[2].append(total_grads_t)
                comp_eff[0].append(self.cur_step)

                # accumulate # of used grads and total calculated grads
                used_grads = used_grads + used_grads_t
                total_grads = total_grads + total_grads_t
        
                lf = open(self._train_dir +"/logfile.txt", "a")
                lf.write('M iter {} nByz {} fCheck {} nFault {} compEff {}/{} lr {} q {}\n'.format(self.cur_step, len(self._byzantine_workers), int(self._q == 1.0 or fault_check), len(faulty_worker_ranks), used_grads_t,total_grads_t, self.optimizer.param_groups[0]['lr'], self._q))
                lf.close()

            # end if self.redundancy
            else:
                gradient_fetch_requests = self.async_fetch_gradient_start()

                while not enough_gradients_received:
                    status = MPI.Status()
                    if self._compress_grad == 'None':
                        MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)
                    elif self._compress_grad == "compress":
                        t, received_msg = MPI.Request.waitany(requests=gradient_fetch_requests, status=status)
                        # print(t)
                        # print("Master just received compressed message in length ", len(received_msg), "tag=",status.tag - 88)
                        received_grad = decompress(received_msg)

                    if status.tag - 88 in self.grad_accumulator.model_index_range:
                        if not self._first_grad_received:
                            self._first_grad_received = True
                            grad_gather_start_time = time.time()

                        layer_index = status.tag - 88

                        if self._compress_grad == "None":
                            received_grad = self.grad_accumulator.gradient_aggregator[layer_index][status.source - 1]
                            # with open("masterlog.txt","a+") as f:
                            #    # f.write("Master received grad in shape: ", type(received_grad), received_grad.shape, " with source=",status.source," and tag=",status.tag-88,"which has the value\n",received_grad,"\n")
                            #    # f.write("The shape should be ", self._model_shapes[layer_index]," for layer idx ",layer_index,"\n")
                            #    f.write("Master received grad in shape: "+str(type(received_grad))+str(received_grad.shape)+
                            #            " with source="+str(status.source)+" and tag="+str(status.tag - 88)+"which has the value\n"+
                            #            str(received_grad)+"\n")
                            #    f.write("The shape should be "+str(self._model_shapes[layer_index])+" for layer idx "+str(layer_index)+
                            #            "\n")
                            #    f.write("The shape used to be "+str(self.grad_accumulator._shape_counter[layer_index][status.source-1]))
                        # check gradient shape
                        # print(received_grad.shape)
                        # print("Received from worker ",status.source-1,": gradient with norm",np.linalg.norm(received_grad.reshape(-1)))
                        assert (received_grad.shape == self._model_shapes[layer_index])

                        # aggregate the gradient
                        if self.grad_accumulator.gradient_aggregate_counter[layer_index] <= self._num_grad_to_collect:
                            # if layer_index < 2:
                            #     print("[{}] worker={} layer={}\t{}".format(self.cur_step,status.source,layer_index,received_grad.tolist()))
                            self.aggregate_gradient(gradient=received_grad, layer_idx=layer_index, source=status.source-1)

                        self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1

                    enough_gradients_received = True
                    for j in self.grad_accumulator.gradient_aggregate_counter:
                        enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)

            if self._err_mode in ['cwtm', 'krum', 'krum2', 'normfilter', 'normfilter2', 'normfilter3']:
                self._err_simulator()

            if self._calculate_cosine and self.cur_step % self._eval_freq == 0:
                self._received_grads = self._grad_aggregate_buffer.copy()

            if self._accumulative == True:
                for g_idx, grads in enumerate(self._grad_aggregate_buffer):
                    if self._accumulative_alpha == 0:
                        self._historical_buffer[g_idx] = self._historical_buffer[g_idx] + np.array(grads)
                        self._grad_aggregate_buffer[g_idx] = self._historical_buffer[g_idx] / self.cur_step
                    else:
                        self._historical_buffer[g_idx] = self._accumulative_alpha * self._historical_buffer[g_idx] + (1-self._accumulative_alpha) * np.array(grads)
                        self._grad_aggregate_buffer[g_idx] = self._historical_buffer[g_idx]

            # update by given gradient filter
            if self._redundancy:
                method_start = time.time()
                # self._avg_received_grads_n(total_grads_t)
                self._avg_received_grads()
                method_duration = time.time() - method_start
            elif self._update_mode == 'normal':
                method_start = time.time()
                self._avg_received_grads()
                method_duration = time.time() - method_start
            elif self._update_mode == 'geometric_median':
                method_start = time.time()
                if self._full_grad == True:
                    self._geo_median()
                else:
                    self._geo_median_splited()
                method_duration = time.time() - method_start
            elif self._update_mode == 'krum':
                method_start = time.time()
                if self._full_grad == True:
                    self._krum()
                else:
                    self._krum_splited()
                method_duration = time.time() - method_start
            elif self._update_mode == 'multi_krum':
                method_start = time.time()
                if self._full_grad == True:
                    self._multi_krum(self._multi_krum_m)
                else:
                    self._multi_krum_splited(self._multi_krum_m)
                method_duration = time.time() - method_start
            elif self._update_mode == 'multi_krum_multi_rounds':
                method_start = time.time()
                if self._full_grad == True:
                    self._multi_krum_multi_rounds(self._multi_krum_m)
                else:
                    self._multi_krum_splited(self._multi_krum_m)
                method_duration = time.time() - method_start
            elif self._update_mode == 'coor_wise_median':
                method_start = time.time()
                self._coor_wise_median()
                method_duration = time.time() - method_start
            elif self._update_mode == 'coor_wise_trimmed_mean':
                method_start = time.time()
                self._coor_wise_trimmed_mean()
                method_duration = time.time() - method_start
            elif self._update_mode == 'median_of_means':
                method_start = time.time()
                if self._full_grad == True:
                    self._median_of_means()
                else:
                    self._median_of_means_splited()
                method_duration = time.time() - method_start
            elif self._update_mode == 'grad_norm':
                method_start = time.time()
                if self._full_grad == True:
                    self._grad_norm_full_grad()
                else:
                    self._grad_norm()
                method_duration = time.time() - method_start
            elif self._update_mode == 'grad_norm_coor_wise':
                method_start = time.time()
                self._grad_norm_coor_wise()
                method_duration = time.time() - method_start
            elif self._update_mode == 'grad_norm_multi_parts':
                method_start = time.time()
                self._grad_norm_multi_parts()
                method_duration = time.time() - method_start
            elif self._update_mode == 'ensemble_normfilter_multikrum':
                method_start = time.time()
                self._ensemble_normfilter_multikrum(self._multi_krum_m)
                method_duration = time.time() - method_start
            elif self._update_mode == 'ensemble_normfilter_cwtm':
                method_start = time.time()
                self._ensemble_normfilter_cwtm()
                method_duration = time.time() - method_start
            elif self._update_mode == 'ensemble_normfilter_medofmeans':
                method_start = time.time()
                self._ensemble_normfilter_medofmeans()
                method_duration = time.time() - method_start

            if self._calculate_cosine and self.cur_step % self._eval_freq == 0:
                self._filtered_grad = self._grad_aggregate_buffer.copy()

            if self._calculate_cosine and self.cur_step % self._eval_freq == 0:
                def concatenate(grads, single_dimension):
                    if single_dimension:
                        print("single dim", np.array(grads[0]).shape)
                        concatenated_gradients = []
                        for idx, grad in enumerate(grads):
                            if idx == 0:
                                concatenated_gradients = np.array(grad)
                            else:
                                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grad)))
                    else:
                        print("multi dim", np.array(grads[0]).shape)
                        concatenated_gradients = []
                        for idx, grad in enumerate(grads):
                            if idx == 0:
                                concatenated_gradients = np.array(grad)
                            else:
                                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grad)), axis=1)
                    return concatenated_gradients

                print("concatenate received grads")
                self._received_grads = concatenate(self._received_grads,False)
                print("concatenate filtered grad")
                self._filtered_grad = concatenate(self._filtered_grad,True)

                angles = []
                norms = []
                filtered_norm = np.linalg.norm(self._filtered_grad)
                for agent_grad in self._received_grads:
                    n1=agent_grad / np.linalg.norm(agent_grad)
                    n2=self._filtered_grad / filtered_norm
                    dot_product = np.dot(n1,n2)
                    angles.append(np.arccos(dot_product))
                    norms.append(np.linalg.norm(agent_grad))

                norms.append(filtered_norm)

                with open(self._train_dir+"angle.csv","a") as f:
                    csv_writer = csv.writer(f, delimiter=',')
                    csv_writer.writerow([self.cur_step]+angles)

                with open(self._train_dir+"norm.csv","a") as f:
                    csv_writer = csv.writer(f, delimiter=',')
                    csv_writer.writerow([self.cur_step]+norms)

            #print("grad aggregate buffer before update",len(self._grad_aggregate_buffer),"\n\t",self._grad_aggregate_buffer)
            update_start = time.time()
            self.optimizer.step(grads=self._grad_aggregate_buffer, mode=self._update_mode)
            update_duration = time.time() - update_start

            self.meset_grad_buffer()
            self.grad_accumulator.meset_everything()

            if self._eval_freq!=0 and self.cur_step % self._eval_freq == 0:
                self._save_model(file_path=self._generate_model_path())
                # torch.save(torch.get_rng_state(), open(self._train_dir+"rng_state_"+str(self.cur_step),"wb"))
                torch.save(self.optimizer.state_dict(), open(self._train_dir+"optim_"+str(self.cur_step),"wb"))
                if self._diminishing_lr == True:
                    torch.save(self.scheduler.state_dict(), open(self._train_dir+"scheduler_"+str(self.cur_step),"wb"))
            print("Master Step: {}, Method Time Cost: {}, Update Time Cost: {}".format(self.cur_step, method_duration,
                                                                                       update_duration))

            # val_loss = self._evaluate_model(validation_loader, verbose=True)
            # print("[DEBUG] model params")
            # for pidx, param in enumerate(self.network.parameters()):
            #     print(param.data)

            ''' DEBUG param-change triggered fault check '''
            if self._targeted:
                current_params=[]
                for pidx, param in enumerate(self.network.parameters()):
                    # print(param.data)
                    current_params.append(np.array(param.data))
                current_params = np.array(current_params)
                if prev_params is None:
                    prev_params = current_params
                    print("first params shape ",current_params.shape)
                    for i in range(len(current_params)):
                        print("current_params[{}].shape".format(i), current_params[i].shape)
                else:
                    # print('param diff')
                    pdiff = current_params - prev_params
                    # print(tmp - prev_params)
                    # print(type(tmp))
                    # print(tmp.shape)
                    prev_params = current_params

                    # get the average difference between parameters
                    param_diffs.append(np.mean([np.mean(_) for _ in np.abs(pdiff)]))

            ''' rollback save state '''
            if self._roll_freq and (self.cur_step==1):
                print("[Master] Save initial state as rollback")
                val_loss = self._evaluate_model(validation_loader,verbose=True)
                self._save_rollback(val_loss)
            elif self._roll_freq and ((self.cur_step%self._roll_freq)==0):
                val_loss = self._evaluate_model(validation_loader)
                if self._rollback['loss'] is None or val_loss < self._rollback['loss']:
                    print("Master step {} replace old rollback".format(self.cur_step))
                    self._save_rollback(val_loss)
                elif val_loss > self._rollback['loss']:
                    # rollback is strictly better than current state
                    self._load_rollback()
                    # print("Master step {} old rollback retained".format(self.cur_step))
            
            # if self._adaptive and self.cur_step > 60 and self.cur_step%20==0:
            if self._adaptive and self.cur_step%20==0:
                self._adapt_q()
                # self._q += .1
                # print("q ==",self._q)

            with open(self._train_dir+"logs-master",'a') as f:
                f.write('{:.8f},{:.8f}\n'.format(method_duration,update_duration))

            '''
            DEBUG MASTER VALIDATION 
            '''
            if self.cur_step%10==0:
                val_loss = self._evaluate_model(validation_loader, verbose=True)
                '''temp'''
                # if val_loss <= .0005:
                    # print("Master {} validation loss {} <= .0005".format(self.cur_step,val_loss))
                    # break

                print("Master {} val_loss {}".format(self.cur_step, val_loss))
            #     # with open("logs-validationloss", 'a') as f:
            #     #     f.write('{} {:.4f}\n'.format(self.cur_step, val_loss))

            self.cur_step += 1

            # ''' DEBUG EVAL '''
            # if self._max_steps - self.cur_step<5:
            #     val_loss = self._evaluate_model(validation_loader)
            #     print("[DEBUG] model params")
            #     for pidx, param in enumerate(self.network.parameters()):
            #         print(param.data)

            # ''' rollback save state '''
            # if self.cur_step ==1:
            #     print("************ Master first step is 1")

            # if (self.cur_step==1) and self._roll_freq>0:
            #     print("[Master] Save initial state as rollback")
            #     val_loss = self._evaluate_model(validation_loader)
            #     self._save_rollback(val_loss)
            # elif self._roll_freq and ((self.cur_step%self._roll_freq)==0):
            #     val_loss = self._evaluate_model(validation_loader)
            #     if self._rollback['loss'] is None or val_loss < self._rollback['loss']:
            #         print("Master step {} replace old rollback".format(self.cur_step))
            #         self._save_rollback(val_loss)
            #     else:
            #         print("Master step {} old rollback retained".format(self.cur_step))


        if self._redundancy:
            # save computational efficiency calculations
            comp_eff[1].append(used_grads)
            comp_eff[2].append(total_grads)
            comp_eff[0].append(self.cur_step)
            comp_eff = np.array(comp_eff)
            np.save(self._train_dir + "comp_eff.npy",comp_eff)        
            print("Master total gradients used: {}/{}".format(used_grads,total_grads))
            print("total rollbacks: {}".format(total_rolls))
            lf = open(self._train_dir +"/logfile.txt", "a")
            lf.write('rb {}'.format(total_rolls))
            # save .npy of parameter diffs
            # param_diffs <-- array of difference in parameter values between steps
            if self._targeted:
                print('param diffs', len(param_diffs))
                np.save(self._train_dir + 'param_diff.npy', np.array(param_diffs))
                print('grad diffs', len(grad_diffs))
                np.save(self._train_dir + 'grad_diff.npy', np.array(grad_diffs))
            
        print('end of execution')

    def init_model_shapes(self):
        for param_idx, param in enumerate(self.network.parameters()):
            self._model_shapes.append(param.size())
            if self._update_mode == 'normal':
                self._grad_aggregate_buffer.append(np.zeros(param.size()))
                if self._accumulative == True:
                    self._historical_buffer.append(np.zeros(param.size()))
            elif self._update_mode in ('geometric_median', 'krum', 'multi_krum', 'multi_krum_multi_rounds', 'coor_wise_median', 'coor_wise_trimmed_mean',
                                       'median_of_means', 'grad_norm', 'grad_norm_coor_wise', 'grad_norm_full_grad',
                                       'grad_norm_multi_parts', 'ensemble_normfilter_multikrum', 'ensemble_normfilter_cwtm', 'ensemble_normfilter_medofmeans'):
                self._grad_aggregate_buffer.append([np.zeros(param.size()).reshape(-1)]*self.num_workers)
                if self._accumulative == True:
                    self._historical_buffer.append(np.array([np.zeros(param.size()).reshape(-1)]*self.num_workers))

    def async_bcast_step(self):
        """
        broadcasting current step to workers
        """
        req_list = []
        print('Master: Broadcasting current step to workers: step={}'.format(self.cur_step))
        for i in range(self.world_size):
            if i != 0 :
                req_list.append(self.comm.isend(self.cur_step, dest=i, tag=10))
        for i in range(len(req_list)):
            req_list[i].wait()

    def bcast_seed(self):
        req_list = []
        man_seed = random.randrange(0,10000)
        print('Master: Broadcasting seed {} to workers'.format(man_seed))
        for i in range(self.world_size):
            if i != 0:
                req_list.append(self.comm.isend(man_seed, dest=i,tag=7))
        for i in range(len(req_list)):
            req_list[i].wait()

    def bcast_rand_nums(self):
        req_list = []
        # np.random.seed(time.localtime().tm_min)
        np.random.seed()
        for i in range(self.world_size):
            if i != 0:
                nums = [randn() for i in range(self._max_steps)]
                req_list.append(self.comm.isend(nums, dest=i,tag=6))
        for i in range(len(req_list)):
            req_list[i].wait()
        

    def async_bcast_layer_weights_async(self):
        request_layers = []
        for layer_idx, layer in enumerate(self.network.parameters()):
            request_workers = []
            layer_to_send = layer.data.numpy().astype(np.float64)
            for i in range(self.world_size):
                if i != 0:
                    req = self.comm.Isend([layer_to_send, MPI.DOUBLE], dest=1, tag=11 + layer_idx)
                    request_workers.append(req)
            request_layers.append(request_workers)
        for req_l in request_layers:
            for req_worker in req_l:
                req_worker.wait()

    def q_decision(self, q) :
        mu = 0.00
        sigma = 1.00

        r = randn()

        quantile = mu + sigma * sqrt(2)*erfinv(2*q-1)

        if quantile >= r :
            return 1
        else:
            return 0

    def is_empty_element(grad_buff):
        res = True
        for g_list in grad_buff:
            if len(g_list) == 0:
                res = False
                break
        return res 

    def async_bcast_redundant_datapoints(self, old_dp_list, datapoints):
        new_dp_list = [[] for _ in range(self.world_size)]      
        new_worker_list = [[] for _ in range(self.batch_size)]
        req_list = []
        avg_size = ceil((len(datapoints)*(self._s))/self.world_size)
        avail = [i for i in range(1,self.world_size)]
        avail = list(set(avail) - set(self._byzantine_workers))

        for dp in datapoints:   
            for i in range(self._s):
                dst = random.choice(avail)
                while dp in new_dp_list[dst] or dp in old_dp_list[dst]: 
                    dst = random.choice(avail)
                new_dp_list[dst].append(dp)
                new_worker_list[dp].append(dst)
        for i in range(self.world_size):
            if i != 0:
                req_list.append(self.comm.isend(new_dp_list[i], dest=i, tag=8))
        for i in range(len(req_list)):
            req_list[i].wait()
        return new_dp_list, new_worker_list

    def async_bcast_datapoints_redundant(self):
        """
        ADD HANDLING OF BYZANTINE WORKERS
        """
        # print(f"Master {self.cur_step} begin sending datapoints")
        dp_list = [[] for _ in range(self.world_size)]      # list of datapoints (indices) for each worker
                                                            # dp_list[i] = datapoints for ith worker
        worker_list = [[] for _ in range(self.batch_size)]  # list of workers (ranks) for each datapoint index 
                                                            # worker_list[i] = ranks of workers for ith datapoint
        req_list = []
        avg_size = ceil((self.batch_size*(self._s + 1))/self.world_size)
        avail = [i for i in range(1,self.world_size)]
        avail = list(set(avail) - set(self._byzantine_workers))
        count = 0
        for dp in range(self.batch_size):   
            for i in range(self._s + 1):
                if count < len(avail):
                    dst = avail[count]
                    count = count + 1
                else :
                    dst = random.choice(avail)
                while dp in dp_list[dst]: 
                    dst = random.choice(avail)

                dp_list[dst].append(dp)
                worker_list[dp].append(dst)
                
                # if len(dp_list[dst]) >= avg_size :
                #    avail.remove(dst)

        for i in range(self.world_size):
            if i != 0:
                req_list.append(self.comm.isend(dp_list[i], dest=i, tag=9))
        for i in range(len(req_list)):
            req_list[i].wait()

        return dp_list, worker_list

    def async_bcast_datapoints_singular(self):
        dp_list = [[] for _ in range(self.world_size)]
        worker_list = [[] for _ in range(self.batch_size)]
        req_list = []
        avg_size = ceil((self.batch_size*(self._s + 1))/self.world_size)
        avail = [i for i in range(1,self.world_size)]
        avail = list(set(avail) - set(self._byzantine_workers))
        count = 0
        for dp in range(self.batch_size):   
                if count < len(avail):
                    dst = avail[count]
                    count = count + 1
                else :
                    dst = random.choice(avail)
                while dp in dp_list[dst]: 
                    dst = random.choice(avail)

                dp_list[dst].append(dp)
                worker_list[dp].append(dst)
                
        for i in range(self.world_size):
            if i != 0:
                req_list.append(self.comm.isend(dp_list[i], dest=i, tag=9))
        for i in range(len(req_list)):
            req_list[i].wait()

        return dp_list, worker_list

    def async_bcast_layer_weights_bcast(self):
        for layer_idx, layer in enumerate(self.network.parameters()):
            #print("broadcasting layer {}".format(layer_idx))
            layer_to_send = layer.data.numpy().astype(np.float64)
            self.comm.Bcast([layer_to_send, MPI.DOUBLE], root=0)

    def fetch_coded_gradients_redundant(self, new_worker_list, old_worker_list):
        requests = []
        numlayers = len(list(enumerate(self.network.parameters())))

        # expand self.coded_buffer for datapoints being re-broadcasted
        for dpidx, dp in enumerate(new_worker_list):
            if dp:
                # append to self.coded_buffer[dpidx] to accomadate the new gradients
                for _ in dp :
                    temp2=[]
                    for layer_idx, param in enumerate(self.network.parameters()):
                        temp2.append(np.zeros(param.size()))
                    self.coded_buffer[dpidx].append(temp2)
                # print(f"added {len(dp)} slots to dp {dpidx}--> {len(old_worker_list[dpidx])} + {len(new_worker_list[dpidx])} == {len(self.coded_buffer[dpidx])}")

        for dpidx, dp in enumerate(new_worker_list):
            if dp:
                for widx, rank in enumerate(dp):
                    # print(f"dp {dpidx}: Worker [{rank}] to self.coded_buffer[{dpidx}][{len(old_worker_list[dpidx])+widx}]")
                    # print(f"\t\t\t {old_worker_list[dpidx]} {new_worker_list[dpidx]}")
                    for layer_idx, layer in enumerate(self.network.parameters()):
                        req = self.comm.Irecv([self.coded_buffer[dpidx][len(old_worker_list[dpidx])+widx][layer_idx], MPI.DOUBLE],
                                                source=rank, tag=88+(dpidx*numlayers)+layer_idx)
                        requests.append(req)
        return requests

    def fetch_coded_gradients_start(self, worker_list):
        # worker_list[i] should be a list of worker ranks receiving datapoint the ith datapoint
        requests = []
        numlayers = len(list(enumerate(self.network.parameters())))
        for dpidx, dp in enumerate(worker_list):
            for widx, rank in enumerate(dp):
                for layer_idx, layer in enumerate(self.network.parameters()):
                    curtag=88+(dpidx*numlayers)+layer_idx
                    req = self.comm.Irecv([self.coded_buffer[dpidx][widx][layer_idx], MPI.DOUBLE], source=rank, tag=88+(dpidx*numlayers)+layer_idx)
                    requests.append(req)
        return requests
    
    def set_coded_buffer(self, worker_list):
        # worker_list[i] should be a list of worker ranks receiving datapoint the ith datapoint
        self.coded_buffer = [None for _ in range(self.batch_size)]

        for dpidx, dp in enumerate(worker_list):
            temp1 = []
            for widx, worker_rank in enumerate(dp):
                temp2 = []
                for param_idx, param in enumerate(self.network.parameters()):
                    temp2.append(np.zeros((param.size())))
                temp1.append(temp2)
            self.coded_buffer[dpidx] = temp1

    def async_fetch_gradient_start(self):
        gradient_fetch_requests = []
        for layer_idx, layer in enumerate(self.network.parameters()):
            for k in range(self._num_grad_to_collect):
                if self._compress_grad == 'compress':
                    req = self.comm.irecv(self.grad_accumulator.gradient_aggregator[layer_idx][k], source=k+1,
                                          tag=88+layer_idx)
                else:
                    req = self.comm.Irecv([self.grad_accumulator.gradient_aggregator[layer_idx][k], MPI.DOUBLE],
                                          source=k+1, tag=88+layer_idx)
                gradient_fetch_requests.append(req)
        return gradient_fetch_requests

    def aggregate_gradient(self, gradient, layer_idx, source):
        if self._update_mode == 'normal':
            self._grad_aggregate_buffer[layer_idx] += gradient
        elif self._update_mode in ("geometric_median", "krum", 'multi_krum', 'multi_krum_multi_rounds', 'coor_wise_median', 'coor_wise_trimmed_mean',
                                   'median_of_means', 'grad_norm', 'grad_norm_coor_wise', 'grad_norm_full_grad',
                                   'grad_norm_multi_parts', 'ensemble_normfilter_multikrum', 'ensemble_normfilter_cwtm', 'ensemble_normfilter_medofmeans'):
            # print(self._grad_aggregate_buffer[layer_idx][source].shape, gradient.shape)
            # print(self._grad_aggregate_buffer[layer_idx][source].dtype, gradient.dtype)
            self._grad_aggregate_buffer[layer_idx][source] = gradient.reshape(-1)
            """
            _shape = gradient.shape
            if len(_shape) == 1:
                self._grad_aggregate_buffer[layer_idx].append(gradient)
            elif len(_shape) > 1:
                self._grad_aggregate_buffer[layer_idx].append(gradient.reshape(-1))  # gradient.reshape((reduce(lambda x, y: x * y, _shape),)))
            """

    def model_update(self, tmp_module):
        new_state_dict = {}
        model_counter_ = 0
        for param_idx, (key_name, param) in enumerate(self.network.state_dict().items()):
            if "running_mean" in key_name or "running_var" in key_name:
                tmp_dict = {key_name: param}
            else:
                assert param.size() == tmp_module[model_counter_].shape
                tmp_dict = {key_name: torch.from_numpy(tmp_module[model_counter_])}
                model_counter_ += 1
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

    def meset_grad_buffer(self):
        for i in range(len(self._grad_aggregate_buffer)):
            if self._update_mode == 'normal':
                self._grad_aggregate_buffer[i] = np.zeros(self._grad_aggregate_buffer[i].shape)
            elif self._update_mode in ("geometric_median", "krum", 'multi_krum', 'multi_krum_multi_rounds', 'coor_wise_median', 'coor_wise_trimmed_mean',
                                       'median_of_means', 'grad_norm', 'grad_norm_coor_wise', 'grad_norm_full_grad',
                                       'grad_norm_multi_parts', 'ensemble_normfilter_multikrum', 'ensemble_normfilter_cwtm', 'ensemble_normfilter_medofmeans'):
                self._grad_aggregate_buffer[i] = [np.zeros(self._grad_aggregate_buffer[i].shape)]*self.num_workers

    def _err_simulator(self):
        print(self._err_mode)
        if self._err_mode == 'cwtm':
            _honest = list(set(range(0,self.num_workers)) - set(self._adversaries[self.cur_step]))

            for g_idx, grads in enumerate(self._grad_aggregate_buffer):
                coor_wise_sorted = np.sort(np.array(grads)[_honest], axis=0)
                fault_gradient = coor_wise_sorted[min(self._s, len(_honest)-1)]
                for i in self._adversaries[self.cur_step]:
                    self._grad_aggregate_buffer[g_idx][i-1] = fault_gradient
            print(self._err_mode,"err sim finished")
        if self._err_mode == 'krum':
            _honest = list(set(range(0,self.num_workers)) - set(self._adversaries[self.cur_step]))

            concatenated_gradients = None
            separator = []
            #print('concatenation')
            for g_idx, grads in enumerate(self._grad_aggregate_buffer):
                #print('#',g_idx,':',np.array(grads).shape)
                if g_idx == 0:
                    concatenated_gradients = np.array(grads)[_honest]
                else:
                    concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)[_honest]), axis=1)
                separator.append(len(concatenated_gradients[0]))

            def __krum(grad_list, s):
                """
                Krum function in https://arxiv.org/abs/1703.02757
                :param grad_list: gradients from all workers
                :param s: number of faulty workers
                :return: gradient from worker i that minimizes Krum score
                """
                score = []
                for i, g_i in enumerate(grad_list):
                    neighbor_distances = []
                    for j, g_j in enumerate(grad_list):
                        if i!=j:
                            neighbor_distances.append(np.linalg.norm(g_i-g_j)**2)
                    score.append(sum(np.sort(neighbor_distances)[0:self.num_workers-2]))
                i_star = score.index(min(score))
                return grad_list[i_star]

            krum_median = __krum(concatenated_gradients, self._s)
            fault_gradient = np.split(-krum_median,separator[:len(separator)-1])

            for g_idx, grads in enumerate(self._grad_aggregate_buffer):
                for i in self._adversaries[self.cur_step]:
                    self._grad_aggregate_buffer[g_idx][i-1] = fault_gradient[g_idx]
            print(self._err_mode,"err sim finished")
        if self._err_mode == 'krum2':
            _honest = list(set(range(0,self.num_workers)) - set(self._adversaries[self.cur_step]))

            concatenated_gradients = None
            separator = []
            #print('concatenation')
            for g_idx, grads in enumerate(self._grad_aggregate_buffer):
                #print('#',g_idx,':',np.array(grads).shape)
                if g_idx == 0:
                    concatenated_gradients = np.array(grads)[_honest]
                else:
                    concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)[_honest]), axis=1)
                separator.append(len(concatenated_gradients[0]))

            def __krum(grad_list, s):
                """
                Krum function in https://arxiv.org/abs/1703.02757
                :param grad_list: gradients from all workers
                :param s: number of faulty workers
                :return: gradient from worker i that minimizes Krum score
                """
                score = []
                for i, g_i in enumerate(grad_list):
                    neighbor_distances = []
                    for j, g_j in enumerate(grad_list):
                        if i!=j:
                            neighbor_distances.append(np.linalg.norm(g_i-g_j)**2)
                    score.append(sum(np.sort(neighbor_distances)[0:self.num_workers-2]))
                i_star = score.index(max(score))
                return grad_list[i_star]

            worst_krum_median = __krum(concatenated_gradients, self._s)
            fault_gradient = np.split(worst_krum_median,separator[:len(separator)-1])

            for g_idx, grads in enumerate(self._grad_aggregate_buffer):
                for i in self._adversaries[self.cur_step]:
                    self._grad_aggregate_buffer[g_idx][i-1] = fault_gradient[g_idx]
            print(self._err_mode,"err sim finished")
        if self._err_mode == 'normfilter':
            # Reversing direction and being the (n-f) largest norm
            _honest = list(set(range(0,self.num_workers)) - set(self._adversaries[self.cur_step]))

            concatenated_gradients = None
            separator = []
            #print('concatenation')
            for g_idx, grads in enumerate(self._grad_aggregate_buffer):
                #print('#',g_idx,':',np.array(grads).shape)
                if g_idx == 0:
                    concatenated_gradients = np.array(grads)
                else:
                    concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
                separator.append(len(concatenated_gradients[0]))

            fault_norm = np.sort(np.linalg.norm(concatenated_gradients[_honest], axis=1))[max(0,len(_honest)-self._s-1)]

            # note that reverse direction is done in err_simulation() in worker.py. Only need to adjust norm here.
            for i in self._adversaries[self.cur_step]:
                fault_gradient = np.split(concatenated_gradients[i-1] * fault_norm / np.linalg.norm(concatenated_gradients[i-1]), separator[:len(separator)-1])
                for g_idx in range(len(self._grad_aggregate_buffer)):
                    self._grad_aggregate_buffer[g_idx][i-1] = fault_gradient[g_idx]
            print(self._err_mode,"err sim finished")
        if self._err_mode == 'normfilter2':
            # Reversing direction of honest average and being the (n-f) largest norm
            _honest = list(set(range(0,self.num_workers)) - set(self._adversaries[self.cur_step]))

            concatenated_gradients = None
            separator = []
            #print('concatenation')
            for g_idx, grads in enumerate(self._grad_aggregate_buffer):
                #print('#',g_idx,':',np.array(grads).shape)
                if g_idx == 0:
                    concatenated_gradients = np.array(grads)
                else:
                    concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
                separator.append(len(concatenated_gradients[0]))

            fault_norm = np.sort(np.linalg.norm(concatenated_gradients[_honest], axis=1))[max(0,len(_honest)-self._s-1)]
            fault_gradient = np.sum(concatenated_gradients[_honest], axis=0) / len(_honest)
            fault_gradient = np.split(fault_gradient * fault_norm / np.linalg.norm(fault_gradient), separator[:len(separator)-1])

            # note that reverse direction is done in err_simulation() in worker.py. Only need to adjust norm here.
            for i in self._adversaries[self.cur_step]:
                for g_idx in range(len(self._grad_aggregate_buffer)):
                    self._grad_aggregate_buffer[g_idx][i-1] = fault_gradient[g_idx]
            print(self._err_mode,"err sim finished")
        if self._err_mode == 'normfilter3':
            # Reversing direction and being the largest norm
            _honest = list(set(range(0,self.num_workers)) - set(self._adversaries[self.cur_step]))

            concatenated_gradients = None
            separator = []
            #print('concatenation')
            for g_idx, grads in enumerate(self._grad_aggregate_buffer):
                #print('#',g_idx,':',np.array(grads).shape)
                if g_idx == 0:
                    concatenated_gradients = np.array(grads)
                else:
                    concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
                separator.append(len(concatenated_gradients[0]))

            fault_norm = np.sort(np.linalg.norm(concatenated_gradients[_honest], axis=1))[max(0,len(_honest))]

            # note that reverse direction is done in err_simulation() in worker.py. Only need to adjust norm here.
            for i in self._adversaries[self.cur_step]:
                fault_gradient = np.split(concatenated_gradients[i-1] * fault_norm / np.linalg.norm(concatenated_gradients[i-1]), separator[:len(separator)-1])
                for g_idx in range(len(self._grad_aggregate_buffer)):
                    self._grad_aggregate_buffer[g_idx][i-1] = fault_gradient[g_idx]
            print(self._err_mode,"err sim finished")

    def _save_rollback(self, loss):
        # store the rollback params and loss in a dict
        print("[DEBUG] Saving rollback with loss",loss)
        with open('rollback', "wb") as f_:
            roll_dict = {'params': self.network.state_dict(), 'loss': loss}
            # torch.save(self.network.state_dict(), f_)
            torch.save(roll_dict, f_)
        # self._rollback['params'] = self.network.state_dict()
        self._rollback['loss'] = loss

    def _load_rollback(self):
        with open('rollback', "rb") as f_:
            model_state_dict = torch.load(f_)
            print("Loading old rollback with loss",model_state_dict['loss'])
            self.network.load_state_dict(model_state_dict['params'])
        # model_state_dict = self._rollback['params']
        # self.network.load_state_dict(model_state_dict)

    def _generate_model_path(self):
        return self._train_dir + "model_step_" + str(self.cur_step)

    def _save_model(self, file_path):
        with open(file_path, "wb") as f_:
            torch.save(self.network.state_dict(), f_)

    def _load_model(self, file_path):
        with open(file_path, "rb") as f_:
            model_state_dict = torch.load(f_)
            self.network.load_state_dict(model_state_dict)
            print("Master loading checkpoint from {}".format(file_path))

    def _adapt_q(self):
        ''' simplest adaptation -- increase q by .1 '''
        if self._q <= .96:
            self._q += .04
        print("[DEBUG] q ==",self._q)

    def _evaluate_model(self, validation_loader, verbose=False):
        strt = time.time()
        self.network.eval()
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        losses = []
        # while validation_loader.dataset.epochs_completed <= self._epoch_counter:
        for batch_idx,(eval_input_batch,eval_label_batch) in enumerate(validation_loader):
            # eval_input_batch, eval_label_batch = validation_loader.next_batch(batch_size=self._eval_batch_size)
            X_batch, y_batch = Variable(eval_input_batch.float()), Variable(eval_label_batch.float())
            # X_batch, y_batch = Variable(eval_input_batch.float()), Variable(eval_label_batch.long())
            output = self.network(X_batch)
            # if batch_idx == 0: print("LOSS ",self.validation_criterion(output, y_batch).item())
            # losses.append(self.validation_criterion(output, y_batch.long()).item())
            # losses.append(self.validation_criterion(output, y_batch).item())

            ''' temporary for Linreg tests-- dimensions issue with linreg set? '''
            if self.network_config != 'LinregTest':
                losses.append(self.validation_criterion(output, y_batch.long()).item())
                prec1_tmp, prec5_tmp = accuracy(output.data, eval_label_batch.long(), topk=(1,5))
                prec1_counter_ += prec1_tmp
                prec5_counter_ += prec5_tmp
            else :
                losses.append(self.validation_criterion(output, y_batch).item())
                # if batch_idx == 1: print(np.stack([np.array(output.cpu().detach()),np.array(y_batch)],1))
                prec1_counter_ += 0
                prec5_counter_ += 0
                batch_counter_ += 1

        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        # self._epoch_counter = validation_loader.dataset.epochs_completed
        if verbose:
            print('Master {} validation set performance: \tPrec@1: {}\tPrec@5: {}\tAvg Loss: {}\t({:.4f})'.format(self.cur_step, prec1, prec5, np.mean(losses),time.time()-strt))
        return np.mean(losses)

    def _avg_received_grads(self):
        # print("[{}] avg over {}".format(self.cur_step, self._num_grad_to_collect))
        for i in range(len(self._grad_aggregate_buffer)):
            self._grad_aggregate_buffer[i] /= self._num_grad_to_collect

    def _avg_received_grads_n(self, n):
        # print("\t\taveraging",n,"gradients...")
        if n > 0:
            for i in range(len(self._grad_aggregate_buffer)):
                self._grad_aggregate_buffer[i] /= n

    def _redundancy_filter(self):
        for i, grad in enumerate(self._grad_aggregate_buffer):
            print(f"grad #{i}\t{grad.tolist()}")

    def _geo_median(self):
        geo_median_start = time.time()

        concatenated_gradients = None
        separator = []
        #print('concatenation')
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            #print('#',g_idx,':',np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))

        aggregation_finish_time = time.time()

        geo_median = np.array(hd.geomedian(np.array(concatenated_gradients), axis=0))

        filter_finish_time = time.time()
        self._grad_aggregate_buffer = np.split(geo_median,separator[:len(separator)-1])

        print("Master Step: {} Concatenation Cost: {:.4f} Found Geo Median Cost: {:.4f} Splitting Cost: {:.4f}".format(self.cur_step, aggregation_finish_time-geo_median_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))
        with open(self._train_dir+"logs-master",'a') as f:
            f.write('{:.8f},{:.8f},{:.8f},'.format(aggregation_finish_time-geo_median_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))

    def _geo_median_splited(self):
        geo_median_start = time.time()
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            geo_median = np.array(hd.geomedian(np.array(grads), axis=0))
            self._grad_aggregate_buffer[g_idx] = geo_median
        print("Master Step: {} Found Geo Median Cost: {:.4f}".format(self.cur_step, time.time()-geo_median_start))

    def _krum(self):
        # Concatenate parts of krum gradients into a vector;
        # Then Calculate Krum function accordingly, choose the gradient;
        # Finally, break down the gradient into original piece that fits the model
        krum_start = time.time()

        concatenated_gradients = None
        separator = []
        #print('concatenation')
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            #print('#',g_idx,':',np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))

        def __krum(grad_list, s):
            """
            Krum function in https://arxiv.org/abs/1703.02757
            :param grad_list: gradients from all workers
            :param s: number of faulty workers
            :return: gradient from worker i that minimizes Krum score
            """
            score = []
            for i, g_i in enumerate(grad_list):
                neighbor_distances = []
                for j, g_j in enumerate(grad_list):
                    if i!=j:
                        neighbor_distances.append(np.linalg.norm(g_i-g_j)**2)
                score.append(sum(np.sort(neighbor_distances)[0:self.num_workers-s-2]))
            i_star = score.index(min(score))
            return grad_list[i_star]
        
        krum_median = __krum(concatenated_gradients, self._t)
        self._grad_aggregate_buffer = np.split(krum_median,separator[:len(separator)-1])

        print("Master Step: {} Krum Cost: {:.4f}".format(self.cur_step, time.time()-krum_start))


    def _multi_krum_multi_rounds(self, m=1):
        # Concatenate parts of krum gradients into a vector;
        # Then Calculate Krum function accordingly, choose the gradient;
        # Finally, break down the gradient into original piece that fits the model
        krum_start = time.time()

        concatenated_gradients = None
        separator = []
        #print('concatenation')
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            #print('#',g_idx,':',np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))
        aggregation_finish_time = time.time()

        def __krum(grad_list, grad_idxs, s):
            # Krum function.
            #:param grad_list: gradients from all workers
            #:param grad_idxs: list of indexes under consideration
            #:param s: number of faulty workers
            # return: i, gradient from worker i that minimizes Krum score
            score = []
            for i, idx_i in enumerate(grad_idxs):
                neighbor_distances = []
                for j, idx_j in enumerate(grad_idxs):
                    if i!=j:
                        neighbor_distances.append(np.linalg.norm(grad_list[idx_i]-grad_list[idx_j])**2)
                score.append(sum(np.sort(neighbor_distances)[0:self.num_workers-s-2]))
            i_star = score.index(min(score))
            return grad_idxs[i_star], grad_list[grad_idxs[i_star]]

        grads_in_consideration = []
        current_list = list(range(self.num_workers))
        for rnd in range(m):
            print("Round:",rnd)
            i, grad = __krum(concatenated_gradients, current_list, self._t)
            grads_in_consideration.append(grad)
            current_list.remove(i)
        multi_krum_median = np.mean(np.array(grads_in_consideration), axis=0)
        filter_finish_time = time.time()

        self._grad_aggregate_buffer = np.split(multi_krum_median,separator[:len(separator)-1])

        print("Master Step: {} Concatenation Cost: {:.4f} Filter Cost: {:.4f} Splitting Cost: {:.4f}".format(self.cur_step, aggregation_finish_time-krum_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))

    def _multi_krum(self, m=1):
        # Concatenate parts of krum gradients into a vector;
        # Then Calculate Krum function accordingly, choose the gradient;
        # Finally, break down the gradient into original piece that fits the model
        krum_start = time.time()

        concatenated_gradients = None
        separator = []
        #print('concatenation')
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            #print('#',g_idx,':',np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))
        aggregation_finish_time = time.time()

        def __krum(grad_list, s):
            """
            Krum function in https://arxiv.org/abs/1703.02757
            :param grad_list: gradients from all workers
            :param s: number of faulty workers
            :return: gradient from worker i that minimizes Krum score
            """
            score = []
            for i, g_i in enumerate(grad_list):
                neighbor_distances = []
                for j, g_j in enumerate(grad_list):
                    if i!=j:
                        neighbor_distances.append(np.linalg.norm(g_i-g_j)**2)
                score.append(sum(np.sort(neighbor_distances)[0:self.num_workers-s-2]))
            selected_idx = np.argsort(score)[:m]
            return grad_list[selected_idx]
        
        krum_median = np.mean(__krum(concatenated_gradients, self._t), axis=0)
        filter_finish_time = time.time()

        self._grad_aggregate_buffer = np.split(krum_median,separator[:len(separator)-1])

        print("Master Step: {} Concatenation Cost: {:.4f} Filter Cost: {:.4f} Splitting Cost: {:.4f}".format(self.cur_step, aggregation_finish_time-krum_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))
        with open(self._train_dir+"logs-master",'a') as f:
            f.write('{:.8f},{:.8f},{:.8f},'.format(aggregation_finish_time-krum_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))

    def _krum_splited(self):
        # The version trivially treat different parts of gradients separately
        def __krum(grad_list, s):
            """
            Krum function in https://arxiv.org/abs/1703.02757
            :param grad_list: gradients from all workers
            :param s: number of faulty workers
            :return: gradient from worker i that minimizes Krum score
            """
            score = []
            for i, g_i in enumerate(grad_list):
                neighbor_distances = []
                for j, g_j in enumerate(grad_list):
                    if i!=j:
                        neighbor_distances.append(np.linalg.norm(g_i-g_j)**2)
                score.append(sum(np.sort(neighbor_distances)[0:self.num_workers-s-2]))
            i_star = score.index(min(score))
            return grad_list[i_star]
        krum_start = time.time()
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            krum_median = __krum(grads, self._t)
            self._grad_aggregate_buffer[g_idx] = krum_median
        print("Master Step: {} Krum Cost: {:.4f}".format(self.cur_step, time.time()-krum_start))

    def _multi_krum_splited(self, m):
        # The version trivially treat different parts of gradients separately
        def __krum(grad_list, grad_idxs, s):
            """
            Krum function.
            :param grad_list: gradients from all workers
            :param grad_idxs: list of indexes under consideration
            :param s: number of faulty workers
            :return: i, gradient from worker i that minimizes Krum score
            """
            score = []
            for i, idx_i in enumerate(grad_idxs):
                neighbor_distances = []
                for j, idx_j in enumerate(grad_idxs):
                    if i!=j:
                        neighbor_distances.append(np.linalg.norm(grad_list[idx_i]-grad_list[idx_j])**2)
                score.append(sum(np.sort(neighbor_distances)[0:self.num_workers-s-2]))
            i_star = score.index(min(score))
            return grad_idxs[i_star], grad_list[grad_idxs[i_star]]
        krum_start = time.time()
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            grads_in_consideration = []
            current_list = list(range(self.num_workers))
            for rnd in range(m):
                print("Round:",rnd)
                i, grad = __krum(grads, current_list, self._t)
                grads_in_consideration.append(grad)
                current_list.remove(i)
            self._grad_aggregate_buffer[g_idx] = np.mean(np.array(grads_in_consideration), axis=0)
        print("Master Step: {} Multi-Krum cost: {:.4f}".format(self.cur_step, time.time()-krum_start))

    def _coor_wise_median(self):
        cw_median_start = time.time()
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            median = np.median(np.array(grads), axis=0)
            self._grad_aggregate_buffer[g_idx] = median
        print("Master Step: {} Coor wise median cost: {:.4f}".format(self.cur_step, time.time()-cw_median_start))
        with open(self._train_dir+"logs-master",'a') as f:
            f.write('{:.8f},'.format(time.time()-cw_median_start))

    def _coor_wise_trimmed_mean(self):
        cwtm_start = time.time()
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            trimmed_mean = np.mean(np.sort(np.array(grads), axis=0)[self._t:self.num_workers-self._t], axis=0)
            self._grad_aggregate_buffer[g_idx] = trimmed_mean
        print("Master Step: {} Coor wise trimmed mean cost: {:.4f}".format(self.cur_step, time.time()-cwtm_start))
        with open(self._train_dir+"logs-master",'a') as f:
            f.write('{:.8f},'.format(time.time()-cwtm_start))

    """
    def _median_of_means(self):
        b = math.ceil(self.num_workers / (2*self._t+0.5))
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            median = np.median(np.array([np.mean(np.array(grads[i:i+b]), axis=0) for i in range(0,self.num_workers,b)]), axis=0)
            self._grad_aggregate_buffer[g_idx] = median
    """

    def _median_of_means(self):
        medofmeans_start = time.time()
        concatenated_gradients = None
        separator = []
        #print('concatenation')
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            #print('#',g_idx,':',np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))
        aggregation_finish_time = time.time()

        b = math.ceil(self.num_workers / (2*self._t+0.5))

        median = np.array(hd.geomedian(np.array([np.mean(np.array(concatenated_gradients[i:i+b]), axis=0) for i in range(0,self.num_workers,b)]), axis=0))
        filter_finish_time = time.time()

        self._grad_aggregate_buffer = np.split(median,separator[:len(separator)-1])
        print("Master Step: {} Concatenation Cost: {:.4f} Filter Cost: {:.4f} Splitting Cost: {:.4f}".format(self.cur_step, aggregation_finish_time-medofmeans_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))
        with open(self._train_dir+"logs-master",'a') as f:
            f.write('{:.8f},{:.8f},{:.8f},'.format(aggregation_finish_time-medofmeans_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))

    def _median_of_means_splited(self):
        b = math.ceil(self.num_workers / (2*self._t+0.5))
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            median = np.array(hd.geomedian(np.array([np.mean(np.array(grads[i:i+b]), axis=0) for i in range(0,self.num_workers,b)]), axis=0))
            self._grad_aggregate_buffer[g_idx] = median

    def _grad_norm(self):
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            ranks = np.argsort(np.linalg.norm(np.array(grads), axis=1))
            norm = np.linalg.norm(grads[ranks[self.num_workers-self._t-1]])
            for i in range(self.num_workers-self._t, self.num_workers):
                grads[ranks[i]]=grads[ranks[i]]*norm/np.linalg.norm(grads[ranks[i]])
            if self._grad_norm_keep_all == True:
                self._grad_aggregate_buffer[g_idx] = np.sum(np.array(grads), axis=0)/self.num_workers
            else:
                self._grad_aggregate_buffer[g_idx] = np.sum(np.array(grads)[ranks[:(self.num_workers-self._t)]], axis=0)/(self.num_workers-self._t)

    def _grad_norm_coor_wise(self):
        print("size of buffer",len(self._grad_aggregate_buffer))
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            """
            grads: nums_of_workers x size_of_param
            """
            print(g_idx)
            calculated_grad = grads[0]
            ranks = np.argsort(grads, axis=0)
            for i in range(len(grads[0])):
                norm = np.abs(grads[ranks[self.num_workers-self._t-1][i]][i])
                for j in range(self.num_workers-self._t, self.num_workers):
                    grads[ranks[j][i]][i] = grads[ranks[j][i]][i]*norm/np.abs(grads[ranks[j][i]][i])
                summation = 0
                for j in range(self.num_workers):
                    summation += grads[j][i]
                calculated_grad[i] = summation/self.num_workers
            self._grad_aggregate_buffer[g_idx] = calculated_grad

    def _grad_norm_multi_parts(self):
        concatenated_gradients = None
        separator = []
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            print(np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))
        gradient_parts = np.split(concatenated_gradients, list(range(0,len(concatenated_gradients[0]),int(len(concatenated_gradients[0])/self._grad_norm_clip_n)))[1:], axis=1)
        for g_idx, grads in enumerate(gradient_parts):
            print(np.array(grads).shape)
            ranks = np.argsort(np.linalg.norm(np.array(grads), axis=1))
            norm = np.linalg.norm(grads[ranks[self.num_workers-self._t-1]])
            for i in range(self.num_workers-self._t, self.num_workers):
                grads[ranks[i]]=grads[ranks[i]]*norm/np.linalg.norm(grads[ranks[i]])
            if self._grad_norm_keep_all == True:
                gradient_parts[g_idx] = np.sum(np.array(grads), axis=0)/self.num_workers
            else:
                gradient_parts[g_idx] = np.sum(np.array(grads)[ranks[:(self.num_workers-self._t)]], axis=0)/(self.num_workers-self._t)
        concatenated_gradients = None
        for g_idx, grad in enumerate(gradient_parts):
            print(np.array(grad).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grad)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, grad))
        self._grad_aggregate_buffer = np.split(concatenated_gradients,separator[:len(separator)-1])

    def _grad_norm_full_grad(self):
        norm_filter_start = time.time()
        concatenated_gradients = None
        separator = []
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            #print(np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))
        aggregation_finish_time = time.time()
        # print(concatenated_gradients.shape)
        # print(separator)
        ranks = np.argsort(np.linalg.norm(np.array(concatenated_gradients), axis=1))
        #print(np.sqrt(np.sum(np.square([np.linalg.norm(self._grad_aggregate_buffer[i], axis=1) for i in range(len(self._grad_aggregate_buffer))]), axis=0)))
        #print(np.linalg.norm(concatenated_gradients, axis=1))
        #print(np.mean(np.linalg.norm(concatenated_gradients, axis=1)))
        #print(np.linalg.norm(np.mean(concatenated_gradients, axis=0)))

        if self._grad_norm_keep_all == True:
            norm = np.linalg.norm(concatenated_gradients[ranks[self.num_workers-self._t-1]])
            for i in range(self.num_workers-self._t, self.num_workers):
                concatenated_gradients[ranks[i]] = concatenated_gradients[ranks[i]]*norm/np.linalg.norm(concatenated_gradients[ranks[i]])
            #print(np.linalg.norm(concatenated_gradients, axis=1))
            #print(concatenated_gradients[0].shape)
            sum_gradient = np.mean(concatenated_gradients, axis=0)
        else:
            # print(ranks[:(self.num_workers-self._t)])
            sum_gradient = np.mean(np.array(concatenated_gradients)[ranks[:(self.num_workers-self._t)]], axis=0)
        filter_finish_time = time.time()

        # print(sum_gradient.shape)
        #print(np.linalg.norm(sum_gradient))
        self._grad_aggregate_buffer=np.split(sum_gradient,separator[:len(separator)-1])

        print("Master Step: {} Concatenation Cost: {:.4f} Filter Cost: {:.4f} Splitting Cost: {:.4f}".format(self.cur_step, aggregation_finish_time-norm_filter_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))
        with open(self._train_dir+"logs-master",'a') as f:
            f.write('{:.8f},{:.8f},{:.8f},'.format(aggregation_finish_time-norm_filter_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))

    def _ensemble_normfilter_multikrum(self, m):
        ensemble_filter_start = time.time()
        concatenated_gradients = None
        separator = []
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            #print(np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))
        aggregation_finish_time = time.time()
        # print(concatenated_gradients.shape)
        # print(separator)
        ranks = np.argsort(np.linalg.norm(np.array(concatenated_gradients), axis=1))
        #print(np.sqrt(np.sum(np.square([np.linalg.norm(self._grad_aggregate_buffer[i], axis=1) for i in range(len(self._grad_aggregate_buffer))]), axis=0)))
        #print(np.linalg.norm(concatenated_gradients, axis=1))
        #print(np.mean(np.linalg.norm(concatenated_gradients, axis=1)))
        #print(np.linalg.norm(np.mean(concatenated_gradients, axis=0)))

        filtered_gradients = np.array(concatenated_gradients)[ranks[:(self.num_workers-self._t)]]

        def __krum(grad_list, grad_idxs, s):
            # Krum function.
            #:param grad_list: gradients from all workers
            #:param grad_idxs: list of indexes under consideration
            #:param s: number of faulty workers
            # return: i, gradient from worker i that minimizes Krum score
            score = []
            for i, idx_i in enumerate(grad_idxs):
                neighbor_distances = []
                for j, idx_j in enumerate(grad_idxs):
                    if i!=j:
                        neighbor_distances.append(np.linalg.norm(grad_list[idx_i]-grad_list[idx_j])**2)
                score.append(sum(np.sort(neighbor_distances)[0:self.num_workers-self._t-s-2]))
            i_star = score.index(min(score))
            return grad_idxs[i_star], grad_list[grad_idxs[i_star]]

        grads_in_consideration = []
        current_list = list(range(len(filtered_gradients)))
        for rnd in range(m):
            print("Round:",rnd)
            i, grad = __krum(filtered_gradients, current_list, self._t)
            grads_in_consideration.append(grad)
            current_list.remove(i)
        multi_krum_median = np.mean(np.array(grads_in_consideration), axis=0)
        filter_finish_time = time.time()

        self._grad_aggregate_buffer = np.split(multi_krum_median,separator[:len(separator)-1])

        print("Master Step: {} Concatenation Cost: {:.4f} Filter Cost: {:.4f} Splitting Cost: {:.4f}".format(self.cur_step, aggregation_finish_time-ensemble_filter_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))
        with open(self._train_dir+"logs-master",'a') as f:
            f.write('{:.8f},{:.8f},{:.8f},'.format(aggregation_finish_time-ensemble_filter_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))

    def _ensemble_normfilter_cwtm(self):
        ensemble_filter_start = time.time()
        concatenated_gradients = None
        separator = []
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            #print(np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))
        aggregation_finish_time = time.time()
        # print(concatenated_gradients.shape)
        # print(separator)
        ranks = np.argsort(np.linalg.norm(np.array(concatenated_gradients), axis=1))
        #print(np.sqrt(np.sum(np.square([np.linalg.norm(self._grad_aggregate_buffer[i], axis=1) for i in range(len(self._grad_aggregate_buffer))]), axis=0)))
        #print(np.linalg.norm(concatenated_gradients, axis=1))
        #print(np.mean(np.linalg.norm(concatenated_gradients, axis=1)))
        #print(np.linalg.norm(np.mean(concatenated_gradients, axis=0)))

        filtered_gradients = np.array(concatenated_gradients)[ranks[:(self.num_workers-self._t)]]

        cwtm_start = time.time()
        for g_idx, grads in enumerate(filtered_gradients):
            trimmed_mean = np.mean(np.sort(np.array(grads), axis=0)[self._t:self.num_workers-self._t], axis=0)
            filtered_gradients[g_idx] = trimmed_mean
        filter_finish_time = time.time()

        self._grad_aggregate_buffer = np.split(filtered_gradients,separator[:len(separator)-1])

        print("Master Step: {} Concatenation Cost: {:.4f} Filter Cost: {:.4f} Splitting Cost: {:.4f}".format(self.cur_step, aggregation_finish_time-ensemble_filter_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))
        with open(self._train_dir+"logs-master",'a') as f:
            f.write('{:.8f},{:.8f},{:.8f},'.format(aggregation_finish_time-ensemble_filter_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))

    def _ensemble_normfilter_medofmeans(self):
        ensemble_filter_start = time.time()
        concatenated_gradients = None
        separator = []
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            #print(np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))
        aggregation_finish_time = time.time()
        # print(concatenated_gradients.shape)
        # print(separator)
        ranks = np.argsort(np.linalg.norm(np.array(concatenated_gradients), axis=1))
        #print(np.sqrt(np.sum(np.square([np.linalg.norm(self._grad_aggregate_buffer[i], axis=1) for i in range(len(self._grad_aggregate_buffer))]), axis=0)))
        #print(np.linalg.norm(concatenated_gradients, axis=1))
        #print(np.mean(np.linalg.norm(concatenated_gradients, axis=1)))
        #print(np.linalg.norm(np.mean(concatenated_gradients, axis=0)))

        filtered_gradients = np.array(concatenated_gradients)[ranks[:(self.num_workers-self._t)]]

        b = math.ceil(self.num_workers-self._t / (2*self._t+0.5))

        median = np.array(hd.geomedian(np.array([np.mean(np.array(concatenated_gradients[i:i+b]), axis=0) for i in range(0,self.num_workers-self._t,b)]), axis=0))
        filter_finish_time = time.time()

        self._grad_aggregate_buffer = np.split(median,separator[:len(separator)-1])
        print("Master Step: {} Concatenation Cost: {:.4f} Filter Cost: {:.4f} Splitting Cost: {:.4f}".format(self.cur_step, aggregation_finish_time-ensemble_filter_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))
        with open(self._train_dir+"logs-master",'a') as f:
            f.write('{:.8f},{:.8f},{:.8f},'.format(aggregation_finish_time-ensemble_filter_start, filter_finish_time-aggregation_finish_time, time.time()-filter_finish_time))


class GradientAccumulator(object):
    """
    Gradient accumulator like conditionalAccumulator in tensorflow
    """

    def __init__(self, module, num_worker, mode='None'):
        self.gradient_aggregate_counter = []
        self.model_index_range = []
        self.gradient_aggregator = []
        self._mode = mode
        self._shape_counter = []

        #print("Creating receivers: ")
        for param_idx, param in enumerate(module.parameters()):
            tmp_aggregator = []
            tmp_shape_counter = []
            #print("param id:",param_idx)
            for worker_idx in range(num_worker):
                if self._mode == 'None':
                    tmp_aggregator.append(np.zeros((param.size())))
                    tmp_shape_counter.append(tmp_aggregator[worker_idx].shape)
                    #print("adding a layer of size",param.size(),"for worker",worker_idx,tmp_aggregator[worker_idx].shape)
                elif self._mode == 'compress':
                    _shape = param.size()
                    if len(_shape) == 1:
                        tmp_aggregator.append(bytearray(getsizeof(np.zeros((_shape[0],))) * 2))
                    else:
                        tmp_aggregator.append(bytearray(getsizeof(np.zeros(_shape)) * 2))
            self.gradient_aggregator.append(tmp_aggregator)
            self.gradient_aggregate_counter.append(0)
            self.model_index_range.append(param_idx)
            self._shape_counter.append(tmp_shape_counter)
            #print(len(self.gradient_aggregator),self.gradient_aggregator[param_idx][0].shape)

        #print()
        #for param_idx, param in enumerate(module.parameters()):
        #    for worker_idx in range(num_worker):
        #        print("({},{})".format(param_idx,worker_idx),self._shape_counter[param_idx][worker_idx],self.gradient_aggregator[param_idx][worker_idx].shape, param.size())

    def meset_everything(self):
        self._meset_grad_counter()
        self._meset_grad_aggregator()

    def _meset_grad_counter(self):
        self.gradient_aggregate_counter = [0 for _ in self.gradient_aggregate_counter]

    def _meset_grad_aggregator(self):
        if self._mode == 'compress':
            pass
        else:
            for i, tmp_aggregator in enumerate(self.gradient_aggregator):
                for j, buf in enumerate(tmp_aggregator):
                    self.gradient_aggregator[i][j] = np.zeros(self.gradient_aggregator[i][j].shape)
