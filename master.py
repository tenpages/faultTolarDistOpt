import time
from sys import getsizeof
import numpy as np
import hdmedians as hd
import torch
from torch.autograd import Variable

from mpi4py import MPI

from compress_gradient import decompress
from model_ops.fc import Full_Connected_Split
from nn_ops import NN_Trainer, accuracy
from optim.sgd_modified import SGDModified

from functools import reduce

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
        self.cur_step = STEP_START_
        self.lr = kwargs['learning_rate']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']

        self._num_grad_to_collect = self.world_size - 1
        self._grad_aggregate_buffer = []
        self._model_shapes = []
        self._first_grad_received = False
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._max_steps = kwargs['max_steps']
        self._update_mode = kwargs['update_mode']
        self._compress_grad = kwargs['compress_grad']
        self._checkpoint_step = kwargs['checkpoint_step']
        self._s = kwargs['worker_fail']
        self._size = kwargs['data_size']

    def build_model(self) :
        # print("building model, self._size ", self._size)
        if self.network_config == "FC":
            self.network = Full_Connected_Split(self._size)

        if self._checkpoint_step != 0:
            file_path = self._train_dir + "model_step_" + str(self._checkpoint_step)
            self._load_model(file_path)
            self.cur_step = int(self._checkpoint_step)

        # gradient accumulator collects gradients from worker nodes
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size - 1, mode=self._compress_grad)
        self.init_model_shapes()
        # optimizer can be others
        self.optimizer = SGDModified(self.network.parameters(), lr=self.lr, momentum=self.momentum)

    def start(self):
        self.async_bcast_step()

        for i in range(1, self._max_steps + 1):
            self.network.train()
            self._first_grad_received = False
            enough_gradients_received = False

            print("Master node is entering step: {}".format(i))

            self.async_bcast_step()

            if self.comm_type == 'Bcast':
                self.async_bcast_layer_weights_bcast()
            elif self.comm_type == 'Async':
                self.async_bcast_layer_weights_async()

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
                    assert (received_grad.shape == self._model_shapes[layer_index])

                    # aggregate the gradient
                    if self.grad_accumulator.gradient_aggregate_counter[layer_index] <= self._num_grad_to_collect:
                        self.aggregate_gradient(gradient=received_grad, layer_idx=layer_index)

                    self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1

                enough_gradients_received = True
                for j in self.grad_accumulator.gradient_aggregate_counter:
                    enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)

            # update by given gradient filter
            if self._update_mode == 'normal':
                method_start = time.time()
                self._avg_received_grads()
                method_duration = time.time() - method_start
            elif self._update_mode == 'geometric_median':
                method_start = time.time()
                self._get_geo_median()
                method_duration = time.time() - method_start
            elif self._update_mode == 'krum':
                method_start = time.time()
                self._krum()
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
                self._median_of_means()
                method_duration = time.time() - method_start
            elif self._update_mode == 'grad_norm':
                method_start = time.time()
                self._grad_norm()
                method_duration = time.time() - method_start
            elif self._update_mode == 'grad_norm_full_grad':
                method_start = time.time()
                self._grad_norm_full_grad()
                method_duration = time.time() - method_start
            elif self._update_mode == 'grad_norm_coor_wise':
                method_start = time.time()
                self._grad_norm_coor_wise()
                method_duration = time.time() - method_start

            update_start = time.time()
            self.optimizer.step(grads=self._grad_aggregate_buffer, mode=self._update_mode)
            update_duration = time.time() - update_start

            self.meset_grad_buffer()
            self.grad_accumulator.meset_everything()

            if self._eval_freq!=0 and self.cur_step % self._eval_freq == 0:
                self._save_model(file_path=self._generate_model_path())
            print("Master Step: {}, Method Time Cost: {}, Update Time Cost: {}".format(self.cur_step, method_duration,
                                                                                       update_duration))
            self.cur_step += 1

    def init_model_shapes(self):
        for param_idx, param in enumerate(self.network.parameters()):
            self._model_shapes.append(param.size())
            if self._update_mode == 'normal':
                self._grad_aggregate_buffer.append(np.zeros(param.size()))
            elif self._update_mode in ('geometric_median', 'krum', 'coor_wise_median', 'coor_wise_trimmed_mean',
                                       'median_of_means', 'grad_norm', 'grad_norm_coor_wise','grad_norm_full_grad'):
                self._grad_aggregate_buffer.append([])

    def async_bcast_step(self):
        """
        broadcasting current step to workers
        """
        req_list = []
        print('Master: Broadcasting current step to workers: step={}'.format(self.cur_step))
        for i in range(self.world_size):
            if i != 0:
                req_list.append(self.comm.isend(self.cur_step, dest=i, tag=10))
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

    def async_bcast_layer_weights_bcast(self):
        for layer_idx, layer in enumerate(self.network.parameters()):
            layer_to_send = layer.data.numpy().astype(np.float64)
            self.comm.Bcast([layer_to_send, MPI.DOUBLE], root=0)

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

    def aggregate_gradient(self, gradient, layer_idx):
        if self._update_mode == 'normal':
            self._grad_aggregate_buffer[layer_idx] += gradient
        elif self._update_mode in ("geometric_median", "krum", 'coor_wise_median', 'coor_wise_trimmed_mean',
                                   'median_of_means', 'grad_norm', 'grad_norm_coor_wise', 'grad_norm_full_grad'):
            _shape = gradient.shape
            if len(_shape) == 1:
                self._grad_aggregate_buffer[layer_idx].append(gradient)
            elif len(_shape) > 1:
                self._grad_aggregate_buffer[layer_idx].append(gradient.reshape((reduce(lambda x, y: x * y, _shape),)))

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
            elif self._update_mode in ("geometric_median", "krum", 'coor_wise_median', 'coor_wise_trimmed_mean',
                                       'median_of_means', 'grad_norm', 'grad_norm_coor_wise', 'grad_norm_full_grad'):
                self._grad_aggregate_buffer[i] = []

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

    def _evaluate_model(self, validation_loader):
        self.network.eval()
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        while validation_loader.dataset.epochs_completed <= self._epoch_counter:
            eval_input_batch, eval_label_batch = validation_loader.next_batch(batch_size=self._eval_batch_size)
            X_batch, y_batch = Variable(eval_input_batch.float()), Variable(eval_label_batch.long())
            output = self.network(X_batch)
            prec1_tmp, prec5_tmp = accuracy(output.data, eval_label_batch.long(), topk=(1, 5))
            prec1_counter_ += prec1_tmp
            prec5_counter_ += prec5_tmp
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        self._epoch_counter = validation_loader.dataset.epochs_completed
        print('Testset performance: Cur step:{}\tPrec@1: {}\tPrec@5: {}'.format(self.cur_step, prec1, prec5))

    def _avg_received_grads(self):
        for i in range(len(self._grad_aggregate_buffer)):
            self._grad_aggregate_buffer[i] /= self._num_grad_to_collect

    def _get_geo_median(self):
        geo_median_start = time.time()
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            geo_median = np.array(hd.geomedian(np.array(grads), axis=0))
            self._grad_aggregate_buffer[g_idx] = geo_median
        print("Master Step: {} Found Geo Median Cost: {:.4f}".format(self.cur_step, time.time()-geo_median_start))

    def _krum(self):
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
            krum_median = __krum(grads, self._s)
            self._grad_aggregate_buffer[g_idx] = krum_median
        print("Master Step: {} Krum Cost: {:.4f}".format(self.cur_step, time.time()-krum_start))

    def _coor_wise_median(self):
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            median = np.median(np.array(grads), axis=0)
            self._grad_aggregate_buffer[g_idx] = median

    def _coor_wise_trimmed_mean(self):
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            trimed_mean = np.mean(np.sort(np.array(grads), axis=0)[self._s:self.num_workers-self._s], axis=0)
            self._grad_aggregate_buffer[g_idx] = trimed_mean

    """
    def _median_of_means(self):
        b = math.floor(self.num_workers / (2*self._s+0.5))
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            median = np.median(np.array([np.mean(np.array(grads[i:i+b]), axis=0) for i in range(0,self.num_workers,b)]), axis=0)
            self._grad_aggregate_buffer[g_idx] = median
    """

    def _median_of_means(self):
        b = math.floor(self.num_workers / (2*self._s+0.5))
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            median = np.array(hd.geomedian(np.array([np.mean(np.array(grads[i:i+b]), axis=0) for i in range(0,self.num_workers,b)]), axis=0))
            self._grad_aggregate_buffer[g_idx] = median

    def _grad_norm(self):
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            ranks = np.argsort(np.linalg.norm(np.array(grads), axis=1))
            norm = np.linalg.norm(grads[ranks[self.num_workers-self._s-1]])
            for i in range(self.num_workers-self._s, self.num_workers):
                grads[ranks[i]]=grads[ranks[i]]*norm/np.linalg.norm(grads[ranks[i]])
            self._grad_aggregate_buffer[g_idx] = np.sum(np.array(grads), axis=0)/self.num_workers

    def _grad_norm_coor_wise(self):
        print("size of buffer",len(self._grad_aggregate_buffer))
        """
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            print(g_idx)
            calculated_grad = grads[0]
            for idx, i in enumerate(np.array(grads).T):
                # print(idx,':',len(i),len(ranks))
                ranks = np.argsort(np.abs(i))
                norm = np.abs(i[ranks[self.num_workers - self._s-1]])
                for j in range(self.num_workers-self._s, self.num_workers):
                    i[ranks[j]] = i[ranks[j]]*norm/np.abs(i[ranks[j]])
                calculated_grad[idx] = np.average(i)
            self._grad_aggregate_buffer[g_idx] = calculated_grad
        """

        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            """
            grads: nums_of_workers x size_of_param
            """
            print(g_idx)
            calculated_grad = grads[0]
            ranks = np.argsort(grads, axis=0)
            for i in range(len(grads[0])):
                norm = np.abs(grads[ranks[self.num_workers-self._s-1][i]][i])
                for j in range(self.num_workers-self._s, self.num_workers):
                    grads[ranks[j][i]][i] = grads[ranks[j][i]][i]*norm/np.abs(grads[ranks[j][i]][i])
                summation = 0
                for j in range(self.num_workers):
                    summation += grads[j][i]
                calculated_grad[i] = summation/self.num_workers
            self._grad_aggregate_buffer[g_idx] = calculated_grad

    def _grad_norm_full_grad(self):
        concatenated_gradients = None
        separator = []
        for g_idx, grads in enumerate(self._grad_aggregate_buffer):
            print(np.array(grads).shape)
            if g_idx == 0:
                concatenated_gradients = np.array(grads)
            else:
                concatenated_gradients = np.concatenate((concatenated_gradients, np.array(grads)), axis=1)
            separator.append(len(concatenated_gradients[0]))
        # print(concatenated_gradients.shape)
        # print(separator)
        ranks = np.argsort(np.linalg.norm(np.array(concatenated_gradients), axis=1))
        norm = np.linalg.norm(concatenated_gradients[ranks[self.num_workers-self._s-1]])
        for i in range(self.num_workers-self._s, self.num_workers):
            concatenated_gradients[ranks[i]] = concatenated_gradients[ranks[i]]*norm/np.linalg.norm(concatenated_gradients[ranks[i]])
        sum_gradient = np.sum(np.array(concatenated_gradients), axis=0)/self.num_workers
        self._grad_aggregate_buffer=np.split(sum_gradient,separator[:len(separator)-1])
        # print(len(self._grad_aggregate_buffer))
        # for i in self._grad_aggregate_buffer:
        #     print(i.shape)


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
