import torch
from torch.nn.modules.loss import *

class HingeLoss(_Loss):
    def __init__(self, margin: float = 0., size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(HingeLoss, self).__init__(size_average=None, reduce=None, reduction=reduction)
        self.margin = margin
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.clamp(1 - input.t() * target, min=0)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction != 'none':
            raise ValueError("HingeLoss cannot be reduced in a way of {}".format(self.reduction))
        return loss
