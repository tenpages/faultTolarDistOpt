import numpy as np
import blosc


def compress(grad):
    assert isinstance(grad, np.ndarray)
    compressed_grad = blosc.pack_array(grad, cname='blosclz')
    return compressed_grad


def decompress(msg):
    # assert isinstance(msg, bytes)
    grad = blosc.unpack_array(msg)
    return grad