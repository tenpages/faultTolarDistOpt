import time
from functools import reduce
import sys

import torch
from mpi4py import MPI
from torch import nn
from torch.autograd import Variable
from statistics import mean

import numpy as np

from compress_gradient import compress
from model_ops.fc import Full_Connected
from model_ops.lenet import LeNet
from model_ops.resnet import ResNet18
from model_ops.resnetn import ResNet18N
from model_ops.vgg import VGG13, VGG16, VGG19
from nn_ops import NN_Trainer

STEP_START_ = 1


class DistributedWorker(NN_Trainer):
    def __init__(self, comm, **kwargs):
        self.comm = comm
        self.world_size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.cur_step = 0
        self.next_step = 0

        self.redundancy = kwargs['redundancy']
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']
        self.lr = kwargs['learning_rate']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self._adversary = kwargs['adversary']
        self._err_mode = kwargs['err_mode']
        self._compress_grad = kwargs['compress_grad']
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._checkpoint_step = kwargs['checkpoint_step']
        self._max_steps = kwargs['max_steps']
        self._total_size = kwargs['total_size']
        self._channel = kwargs['channel']
        self._size = kwargs['1d_size']

        self._layer_cur_step = []
        self._fail_workers = kwargs['adversaries']

    def build_model(self):
        # print("building model, self._size ", self._size)
        if self.network_config == 'FC':
            self.network = Full_Connected(self._total_size)
        elif self.network_config == 'LeNet':
            self.network = LeNet(self._channel, self._size)
        elif self.network_config == 'ResNet18':
            self.network = ResNet18(self._channel)
        elif self.network_config == 'ResNet18N':
            self.network = ResNet18N(self._channel)
        elif self.network_config == 'VGG13':
            self.network = VGG13(self._channel)
        elif self.network_config == 'VGG16':
            self.network = VGG16(self._channel)
        elif self.network_config == 'VGG19':
            self.network = VGG19(self._channel)

        if self._checkpoint_step != 0:
            file_path = self._train_dir + "model_step_" + str(self._checkpoint_step)
            self._load_model(file_path)

        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()

        self.init_recv_buf()

    def train(self, train_loader, test_loader):
        global STEP_START_

        self.sync_fetch_step()
        assert (self.update_step())
        loader_step = 1
        loader_epoch = 0
        if self._checkpoint_step == 0:
            assert (self.cur_step == STEP_START_)
        else:
            assert (self.cur_step == int(self._checkpoint_step) + 1)
            loader_length = len(train_loader)
            if self.rank==1:
                print("Starting from step=",loader_step)
            while loader_step + loader_length < self.cur_step:
                dump = list(train_loader)
                del dump
                loader_step += loader_length
                loader_epoch += 1
                if self.rank == 1:
                    print("Move to step=",loader_step,"epoch=",loader_epoch)

        num_batch_per_epoch = len(train_loader.dataset) / self.batch_size
        batch_idx = -1
        epoch_idx = 0
        epoch_avg_loss = 0
        iteration_last_step = 0
        iter_start_time = 0
        first = True

        print("Worker {}: starting training".format(self.rank))

        flag = True
        for num_epoch in range(loader_epoch, self.max_epochs):
            for batch_idx, (train_input_batch, train_label_batch) in enumerate(train_loader):
                if self.rank == 1:
                    print("batch_id=",batch_idx,"loader_step=",loader_step,"cur_step=",self.cur_step)
                if loader_step<self.cur_step and flag:
                    loader_step += 1
                    if self.rank == 1:
                        print("skipped")
                        """
                        with open("print-dataset-log-with-checkpoint"+str(self._checkpoint_step), "a+") as f:
                            f.write(str(self.cur_step)+": epoch="+str(num_epoch)+", batch_idx="+str(batch_idx)+" SKIPPED\n")
                            f.write(str(train_input_batch)+"\n")
                            f.write(str(train_label_batch)+"\n")
                            f.write("============================\n")
                        """
                    continue
                else:
                    flag = False

                if self.cur_step == self._max_steps:
                    break

                if self.redundancy:
                        dp_list = torch.LongTensor(self.async_bcast_fetch_datapoints())
                        print(f"Worker [{self.rank}] datapoints {dp_list.tolist()}")

                X_batch, y_batch = Variable(train_input_batch), Variable(train_label_batch)
                
                if self.redundancy :
                        X_batch = torch.index_select(train_input_batch,0,dp_list) 
                        y_batch = torch.index_select(train_label_batch,0,dp_list) 
                        X_batch = Variable(X_batch)
                        y_batch = Variable(y_batch)
                        
                while True:
                    self.async_fetch_step()

                    updated = self.update_step()

                    if (not updated) and (not first):
                        continue

                    if self.rank == 1:
                        if updated:
                            print("====== Updated:", "batch_id=",batch_idx,"cur_step=",self.cur_step)
                        else:
                            print("====== Not updated:", "batch_id=",batch_idx,"cur_step=",self.cur_step)
                        """
                        with open("print-dataset-log-with-checkpoint"+str(self._checkpoint_step), "a+") as f:
                            f.write(str(self.cur_step)+": epoch="+str(num_epoch)+", batch_idx="+str(batch_idx)+"\n")
                            f.write(str(train_input_batch)+"\n")
                            f.write(str(train_label_batch)+"\n")
                            f.write("============================\n")
                        """

                    iteration_last_step = time.time() - iter_start_time
                    iter_start_time = time.time()
                    first = False
                    print('Rank of this node: {}, Current step: {}'.format(self.rank, self.cur_step))

                    fetch_weight_start_time = time.time()
                    if self.comm_type == 'Bcast':
                        self.async_fetch_weight_bcast()
                    elif self.comm_type == 'Async':
                        self.async_fetch_weight_async()
                    fetch_weight_duration = time.time() - fetch_weight_start_time

                    """
                    if self.cur_step>=8 and self.rank==1:
                        with open("model_of_agent_"+str(self.rank)+"_at_step_"+str(self.cur_step)+"_"+str(self._checkpoint_step), "wb") as f:
                            torch.save(self.network.state_dict(), f)
                        # self.network.load_state_dict(torch.load("model_of_agent_"+str(self.rank)+"_at_step_"+str(self.cur_step)))
                        with open("x_batch_of_agent_"+str(self.rank)+"_at_step_"+str(self.cur_step)+"_"+str(self._checkpoint_step), "wb") as f:
                            torch.save(X_batch, f)
                        # X_batch = torch.load("x_batch_of_agent_"+str(self.rank)+"_at_step_"+str(self.cur_step))
                        with open("y_batch_of_agent_"+str(self.rank)+"_at_step_"+str(self.cur_step)+"_"+str(self._checkpoint_step), "wb") as f:
                            torch.save(y_batch, f)
                        # y_batch = torch.load("y_batch_of_agent_"+str(self.rank)+"_at_step_"+str(self.cur_step))
                    """

                    self.network.train()
                    self.optimizer.zero_grad()

                   # if self.redundancy and self.rank == 1:
                   #     logit_list = []
                   #     for i in range(len(X_batch)):
                   #         temp = torch.index_select(X_batch,0,torch.tensor([i])) 
                   #         logit_list.append(self.network(temp))
                            
                    loss_list = []
                    forward_start_time = time.time()
                    logits = self.network(X_batch)
                    if self.redundancy:
                        """
                        testing individual loss values and gradients
                        """
                        numlayers = 0
                        for _ in self.network.parameters():
                            numlayers = numlayers + 1
                        send_check_requests=[]
                        for idx, log in enumerate(logits):
                            self.optimizer.zero_grad()
                            temp1 = torch.index_select(logits,0,torch.tensor(idx))
                            temp2 = torch.index_select(y_batch,0,torch.tensor(idx))
                            # loss_list.append(self.criterion(temp1,temp2))
                            loss = self.criterion(temp1,temp2)
                 
                            loss.backward(retain_graph=True)
                            for param_idx, param in enumerate(self.network.parameters()):
                                grad = param.grad.data.numpy().astype(np.float64)
                                # error simulation
                                if self.rank in self._fail_workers[self.cur_step]:
                                    # print(f"Error sent by worker {self.rank}")
                                    grad = err_simulation(grad, self._err_mode)
                                req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+(dp_list[idx]*numlayers)+param_idx)
                                send_check_requests.append(req_isend)
        
                        for req in send_check_requests:
                            req.wait()

                        # all gradients sent
                        # now wait for a new list of datapoints
                        #   if empty, continue 
                        #   else, repeat lines 204-229 for with UNCHANGED network
                        
                        new_dp_list = torch.LongTensor(self.async_bcast_fetch_datapoints())

                        if new_dp_list.tolist():
                            # self.network.train()
                            self.optimizer.zero_grad()
                            print(f"Worker [{self.rank}] redundant datapoints {new_dp_list.tolist()}")

                            # get new inputs/targets the batch
                            send_check_requests=[]

                            X_batch, y_batch = Variable(train_input_batch), Variable(train_label_batch)
                            X_batch = torch.index_select(train_input_batch,0,new_dp_list) 
                            y_batch = torch.index_select(train_label_batch,0,new_dp_list) 
                            X_batch = Variable(X_batch)
                            y_batch = Variable(y_batch)

                            # get new logits and new losses
                            new_logits = self.network(X_batch)

                            # send back gradients of new losses in PARALLEL to new_dp_list
                            for idx, log in enumerate(new_logits):
                                self.optimizer.zero_grad()
                                temp1 = torch.index_select(new_logits,0,torch.tensor(idx))
                                temp2 = torch.index_select(y_batch,0,torch.tensor(idx))
                                loss = self.criterion(temp1,temp2)
                     
                                loss.backward(retain_graph=True)
                                for param_idx, param in enumerate(self.network.parameters()):
                                    grad = param.grad.data.numpy().astype(np.float64)
                                    if self.rank in self._fail_workers[self.cur_step]:
                                        # print(f"Error sent by worker {self.rank}")
                                        grad = err_simulation(grad, self._err_mode)
                                    req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+(new_dp_list[idx]*numlayers)+param_idx)
                                    send_check_requests.append(req_isend)
            
                            for req in send_check_requests:
                                req.wait()

                            # end if new_dp_list
                        else :
                            print(f"Workr {self.rank} received no redundant datapoints")
                        
                        if self.cur_step > 1:
                            sys.exit()
                    
                    elif "FC" in self.network_config:
                        #print("loss calculation", X_batch.shape, logits.shape, y_batch.shape)
                        loss = self.criterion(logits, y_batch)
                        #print(loss)
                    elif "LeNet" in self.network_config:
                        loss = self.criterion(logits, y_batch)
                    elif "ResNet" in self.network_config:
                        loss = self.criterion(logits, y_batch)
                    elif "VGG" in self.network_config:
                        loss = self.criterion(logits, y_batch)
                    else:
                        raise Exception("No such network as "+self.network_config)

                    if not self.redundancy :
                        epoch_avg_loss += loss.item()
                        forward_duration = time.time() - forward_start_time

                        if "FC" in self.network_config:
                            computation_time, c_duration = self._backward(loss, computation_time=forward_duration)
                        elif "LeNet" in self.network_config:
                            computation_time, c_duration = self._backward(loss, computation_time=forward_duration)
                        elif "ResNet" in self.network_config:
                            computation_time, c_duration = self._backward(loss, computation_time=forward_duration)
                        elif "VGG" in self.network_config:
                            computation_time, c_duration = self._backward(loss, computation_time=forward_duration)

                        # prec1, prec3 = accuracy(logits.data, train_label_batch.long(), topk=(1, 3))
                        prec1, prec3 = accuracy(logits.data, y_batch.long(), topk=(1, 3))
                        with open(self._train_dir+"logs-worker-"+str(self.rank), "a") as f:
                            f.write('{:.8f}\n'.format(time.time()-iter_start_time))
                        print(
                            'Worker: {}, Step: {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.8f}, Time Cost: {:.4f}, Comp: {:.4f}, Comm: {:.4f}, Prec@1: {}, Prec@3: {}'.format(
                                self.rank,
                                self.cur_step, num_epoch, batch_idx * self.batch_size, len(train_loader.dataset),
                                (100. * (batch_idx * self.batch_size) / len(train_loader.dataset)), loss.item(),
                                                          time.time() - iter_start_time, computation_time,
                                                          c_duration + fetch_weight_duration,
                                prec1.numpy()[0], prec3.numpy()[0]))

                    if self.cur_step % self._eval_freq == 0 and self.rank == 1:
                        # save snapshots
                        if "FC" in self.network_config:
                            pass
                    break

    def init_recv_buf(self):
        self.model_recv_buf = ModelBuffer(self.network)

    def sync_fetch_step(self):
        self.next_step = self.comm.recv(source=0, tag=10)
        print('Worker {}: Worker {} just received next step syncly: step={}'.format(self.rank, self.rank, self.next_step))

    def async_fetch_step(self):
        req = self.comm.irecv(source=0, tag=10)
        self.next_step = req.wait()
        print('Worker {}: Worker {} just received next step: step={}'.format(self.rank, self.rank, self.next_step))

    def async_bcast_fetch_datapoints(self):
        req = self.comm.irecv(source=0, tag=9)
        datapoints = req.wait()
        return datapoints

    def async_fetch_weight_async(self):
        request_layers = []
        layers_to_update = []
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            if self.model_recv_buf.layer_cur_step[layer_idx] < self.cur_step:
                layers_to_update.append(layer_idx)
                req = self.comm.Irecv([self.model_recv_buf.recv_buf[layer_idx], MPI.DOUBLE], source=0, tag=11+layer_idx)
                request_layers.append(req)

        assert (len(layers_to_update) == len(request_layers))
        weights_to_update = []
        for req_idx, req_l in enumerate(request_layers):
            req_l.wait()
            weights = self.model_recv_buf.recv_buf[req_idx]
            weights_to_update.append(weights)
            self.model_recv_buf.layer_cur_step[req_idx] = self.cur_step
        self.model_update(weights_to_update)

    def async_fetch_weight_bcast(self):
        layers_to_update = []
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            if self.model_recv_buf.layer_cur_step[layer_idx] < self.cur_step:
                layers_to_update.append(layer_idx)
                self.comm.Bcast([self.model_recv_buf.recv_buf[layer_idx], MPI.DOUBLE], root=0)
        weights_to_update = []
        for req_idx, layer_idx in enumerate(layers_to_update):
            weights = self.model_recv_buf.recv_buf[req_idx]
            weights_to_update.append(weights)
            self.model_recv_buf.layer_cur_step[req_idx] = self.cur_step
        self.model_update(weights_to_update)

    def update_step(self):
        changed = (self.cur_step != self.next_step)
        self.cur_step = self.next_step
        return changed

    def model_update(self, weights_to_update):
        new_state_dict = {}
        model_counter_ = 0
        for param_idx, (key_name, param) in enumerate(self.network.state_dict().items()):
            if 'running_mean' in key_name or 'running_var' in key_name or 'num_batches_tracked' in key_name:
                tmp_dict={key_name: param}
            else:
                assert param.size() == weights_to_update[model_counter_].shape
                tmp_dict = {key_name: torch.from_numpy(weights_to_update[model_counter_])}
                model_counter_ += 1
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

    """
    NEEDS ERROR SIMULATION HANDLING
    """
    def _multi_backward(self, losses, datapoints, computation_time=None):
        send_check_requests = []

        b_start = time.time()

        numlayers = 0
        for _ in self.network.parameters():
            numlayers = numlayers + 1
        
        #testing tags for gradient messages
        for idx, loss in enumerate(losses):
            loss.backward(retain_graph=True)
            # self._print_grads()
            for param_idx, param in enumerate(self.network.parameters()):
                grad = param.grad.data.numpy().astype(np.float64)
                curtag=88+(datapoints[idx]*numlayers)+param_idx
                req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+(datapoints[idx]*numlayers)+param_idx)
                send_check_requests.append(req_isend)
        
        b_duration = time.time()-b_start
        
        computation_time += b_duration
        c_start = time.time()

        for req in send_check_requests:
            req.wait()

        c_duration = time.time() - c_start
        # print("done sending grads for all points of worker",self.rank)
        return computation_time, c_duration

    def _backward(self, loss, logits_1=None, computation_time=None):
        #print('in _backward',self.network_config)
        b_start = time.time()

        loss.backward()

        b_duration = time.time()-b_start

        computation_time += b_duration
        c_start = time.time()
        self._send_grads()
        c_duration = time.time() - c_start
        return computation_time, c_duration

    def _print_grads(self):
        grads = []
        for param_idx, param in enumerate(self.network.parameters()):
            grad = param.grad.data.numpy().astype(np.float64)
            grads.append(grad)
        #print(f"[{self.rank}] printing gradients: {grads}")

    def _send_grads(self):
        req_send_check = []
        concatenated = None
        concatenatedWrong = None
        printNorms = []
        for param_idx, param in enumerate(self.network.parameters()):
            grad = param.grad.data.numpy().astype(np.float64)
            # if self.rank == 1:
            #         file = open("gradient_test_redundancy.txt","a")
            #         file.write(f"Worker #{self.rank} gradient #{param_idx} (length {len(grad)}) {type(grad)} {type(grad[0])}\n\t{grad}\n")
            #         file.close()
            """
            print(grad.shape)
            _shape = grad.shape
            if param_idx == 0:
                concatenated = grad.reshape((reduce(lambda x, y: x * y, _shape),))
            else:
                concatenated = np.concatenate((concatenated, grad.reshape((reduce(lambda x, y: x * y, _shape),))))
            """

            if len(req_send_check) != 0:
                req_send_check[-1].wait()
            if self.rank in self._fail_workers[self.cur_step]:
                simulated_grad = err_simulation(grad, self._err_mode)
                """
                if param_idx == 0:
                    concatenatedWrong = simulated_grad.reshape((reduce(lambda x, y: x * y, _shape),))
                else:
                    concatenatedWrong = np.concatenate((concatenatedWrong, simulated_grad.reshape((reduce(lambda x, y: x * y, _shape),))))
                printNorms.append(np.linalg.norm(simulated_grad.reshape(-1)))
                """

                if self._compress_grad == 'compress':
                    _compressed_grad = compress(simulated_grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+param_idx)
                    req_send_check.append(req_isend)
                else:
                    req_isend = self.comm.Isend([simulated_grad, MPI.DOUBLE], dest=0, tag=88+param_idx)
                    req_send_check.append(req_isend)
            else:
                """
                printNorms.append(np.linalg.norm(grad.reshape(-1)))
                """

                if self._compress_grad == 'compress':
                    _compressed_grad = compress(grad)
                    #print(self.rank," is sending gradients to master on step ",self.cur_step," for parameter ",param_idx," in length ",len(_compressed_grad))
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+param_idx)
                    req_send_check.append(req_isend)
                else:
                    #with open("worker{}log.txt".format(self.rank),"a+") as f:
                    #    f.write(str(self.rank)+" is sending grads to master on step "+str(self.cur_step)+" for parameter "+str(param_idx)+" in shape "+str(grad.shape)+"which has the value\n"+str(grad)+"\n")
                    req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+param_idx)
                    req_send_check.append(req_isend)
                """
                if self.rank in self._fail_workers[self.cur_step]:
                    print("Faulty node",self.rank,"sending gradient with norm=",np.linalg.norm(concatenatedWrong),"which should have been",np.linalg.norm(concatenated),
                          "Gradients:",printNorms)
        else:
            print("Normal node",self.rank,"sending gradient with norm=",np.linalg.norm(concatenated),
                  "Gradients:",printNorms)
        """
        # print(f"Worker [{self.rank}] sending {len(req_send_check)} gradients)")
        req_send_check[-1].wait()

    def _load_model(self, file_path):
        with open(file_path, "rb") as f_:
            model_state_dict = torch.load(f_)
            self.network.load_state_dict(model_state_dict)
            print("Validation Worker Done Loading Checkpoint from {}".format(file_path))


def accuracy(output, target, topk=(1,)):
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


class ModelBuffer(object):
    def __init__(self, network):
        self.recv_buf = []
        self.layer_cur_step = []
        for param_idx, param in enumerate(network.parameters()):
            self.recv_buf.append(np.zeros(param.size()))
            self.layer_cur_step.append(0)


def err_simulation(grad, mode, cyclic=False):
    ADVERSARY_ = -100
    CONST_ = -100
    if mode == 'rev_grad':
        if cyclic:
            adv = ADVERSARY_ * grad
            assert adv.shape == grad.shape
            return np.add(adv, grad)
        else:
            return ADVERSARY_ * grad
    elif mode == 'constant':
        if cyclic:
            adv = np.ones(grad.shape, dtype=np.float64)*CONST_
            assert adv.shape == grad.shape
            return np.add(adv,grad)
        else:
            return np.ones(grad.shape, dtype=np.float64)*CONST_
    else:
        if cyclic:
            adv = ADVERSARY_ * grad
            assert adv.shape == grad.shape
            return np.add(adv, grad)
        else:
            return ADVERSARY_ * grad
