"""Random generators (mainly used for testing)"""

from numbers import Number
import numpy as np

from .poly import Polynomial

def random_real(c):
    return c*np.random.rand()

def random_complex(c):
    return c*np.random.rand() + c*1j*np.random.rand()

def random_sequence(c, N):
    if isinstance(N, Number):
        N = (N,)
    
    if len(N) == 1:
        return [random_complex(c) for _ in range(N[0])]
    
    l = []
    for _ in range(N[0]):
        l.append(random_sequence(c, N[1:]))

    return l

def random_polynomial(N, eta):
    b = Polynomial(random_sequence(10000, N))
    
    s = b.sup_norm(4*N)
    if s > eta:
        return b * ((1 - eta) / s)
    return b

def random_real_sequence(c, N):
    return [random_real(c) for _ in range(N)]

def random_real_polynomial(N, eta):
    b = Polynomial(random_real_sequence(10000, N))
    
    s = b.sup_norm(4*N)
    if s > eta:
        return b * ((1 - eta) / s)
    return b

def random_list(c, shape: tuple[int]):
    l = []
    if len(shape) == 1:
        for _ in range(shape[0]):
            l.append(random_complex(c))
        return l
    
    for _ in range(shape[0]):
        l.append(random_list(c, shape[1:]))

    return l