import torch
from torch.optim import Optimizer


class SGDModified(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov)
        if nesterov and (momentum<=0 or dampening!=0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDModified, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDModified, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, grads, mode, closure=None):
        loss=None
        if closure is not None:
            loss=closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i,p in enumerate(group['params']):
                if mode=='normal':
                    d_p=torch.from_numpy(grads[i]).float()
                elif mode=='geometric_median' or mode=='maj_vote' or mode=='cyclic' or mode=='krum' or mode=='multi_krum'\
                        or mode=='multi_krum_multi_rounds'\
                        or mode=='median_of_means' or mode=='grad_norm' or mode=='coor_wise_median'\
                        or mode=='coor_wise_trimmed_mean' or mode=='grad_norm_coor_wise' or mode=='grad_norm_full_grad'\
                        or mode=='grad_norm_multi_parts'\
                        or mode=='ensemble_normfilter_multikrum' or mode=='ensemble_normfilter_cwtm'\
                        or mode=='ensemble_normfilter_medofmeans':
                    d_p=torch.from_numpy(grads[i].reshape(p.size())).float()
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1-dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
        return loss
