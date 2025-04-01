
from numbers import Number
import mpmath as mp

import numerics as bd
from poly import Polynomial


def random_sequence(c, N):
    if isinstance(N, Number):
        N = (N,)
    
    if len(N) == 1:
        return [bd.make_complex(c*mp.rand() + c*1j*mp.rand()) for _ in range(N[0])]
    
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