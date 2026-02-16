
import random
from numbers import Number
import mpmath as mp

from . import numerics as bd
from .poly import Polynomial

def random_real(c):
    return bd.make_float(c*mp.rand())

def random_complex(c):
    return bd.make_complex(c*mp.rand() + c*1j*mp.rand())

def random_sequence(c, N):
    if isinstance(N, Number):
        N = (N,)
    
    if len(N) == 1:
        return [random_complex(c) for _ in range(N[0])]
    
    l = []
    for k in range(N[0]):
        l.append(random_sequence(c, N[1:]))

    return l

def random_polynomial(N, eta):
    b = Polynomial(random_sequence(10000, N))
    
    s = b.sup_norm(4*N)
    if s > eta:
        return b * ((1 - eta) / s)
    return b

def random_real_sequence(c, N):
    return [bd.make_complex(c*mp.rand()) for _ in range(N)]

def random_real_polynomial(N, eta):
    b = Polynomial(random_real_sequence(10000, N))
    
    s = b.sup_norm(4*N)
    if s > eta:
        return b * ((1 - eta) / s)
    return b

def random_stairlike_sequence_2d(c, shape: tuple[int] = None, directions: str = None):
    """directions is a string of either '^' (up) or '>' (right), giving the path of the stairlike sequence.
    Default is taken randomly."""

    if (shape is None and directions is None) or (shape != None and directions != None):
        raise ValueError("Only one of shape and directions should be defined.")
    
    if shape != None:
        if len(shape) != 2:
            raise ValueError("Only 2D shapes are allowed.")
        
        directions = list("^" * (shape[0] - 1) + ">" * (shape[1] - 1))
        random.shuffle(directions)
    else:
        directions = list(directions)

    cur_seq = [random_complex(c)]
    seq = []
    for d in directions:
        if d == '^':
            cur_seq.append(random_complex(c))
        elif d == '>':
            seq.append(cur_seq)
            cur_seq = [random_complex(c)]

    if len(cur_seq) != 0:
        seq.append(cur_seq)

    return seq

def random_list(c, shape: tuple[int]):
    l = []
    if len(shape) == 1:
        for _ in range(shape[0]):
            l.append(random_complex(c))
        return l
    
    for _ in range(shape[0]):
        l.append(random_list(c, shape[1:]))

    return l