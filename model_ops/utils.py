import numpy as np

ADVERSARY_ = -100
CONST_ = -100

def err_simulation(grad, mode, cyclic=False):
    if mode == "rev_grad":
        # reversed gradient
        if cyclic:
            adv = ADVERSARY_ * grad
            assert adv.shape == grad.shape
            return np.add(adv, grad)
        else:
            return ADVERSARY_ * grad
    elif mode == "constant":
        # constant err gradient
        if cyclic:
            adv = np.ones(grad.shape, dtype=np.float64)*CONST_
            assert adv.shape == grad.shape
            return np.add(adv, grad)
        else:
            return np.ones(grad.shape, dtype=np.float64)*CONST_
    elif mode == "random":
        # random gradient
        # TODO: this is not properly implemented
        return grad
