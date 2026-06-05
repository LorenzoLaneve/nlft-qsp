from typing import TypeAlias

import numpy as np
import scipy as sp

from ..util import next_power_of_two


complex_type: TypeAlias = np.complex128
float_type: TypeAlias = np.float64


def machine_eps():
    return np.finfo(complex_type).eps

def machine_threshold():
    return 1e-8 # TODO make tunable


def unitroots(N: int): # TODO move to util
    """Returns a list containing the N-th roots of unity."""
    return [np.exp(2j*np.pi*k/N) for k in range(N)]
    
def matrix(x: list):
    """Constructs a numpy matrix representing a matrix with the given list (of lists) of coefficients."""
    return np.array(x, dtype=complex_type)
    
def zeros(m: int, n: int):
    """Constructs a `m x n` zero matrix."""
    return np.zeros(shape=(m, n), dtype=complex_type)

def eye(n: int):
    """Constructs the `n x n` identity matrix."""
    return np.eye(n, dtype=complex_type)

    
def poly2cheb(p):
    """Returns a list of coefficients corresponding to the
    Chebyshev expansion of the polynomial `p`, given as a python list of coefficients."""
    return np.polynomial.chebyshev.poly2cheb(p).tolist()

def cheb2poly(c):
    """Returns a list of coefficients corresponding to the
    polynomial of the Chebyshev expansion `c`, given as a python list of coefficients."""
    return np.polynomial.chebyshev.cheb2poly(c).tolist()