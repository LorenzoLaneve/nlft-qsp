from typing import TypeAlias

import numpy as np


complex_type: TypeAlias = np.complex128
float_type: TypeAlias = np.float64


def machine_eps():
    return np.finfo(complex_type).eps

def machine_threshold():
    return 1e-8 # TODO make tunable


def unitroots(N: int): # TODO move to util
    r"""Returns a list containing the $N$-th roots of unity."""
    return [np.exp(2j*np.pi*k/N) for k in range(N)]
    
def matrix(x: list):
    r"""Constructs a numpy matrix representing a matrix with the given list (of lists) of coefficients."""
    return np.array(x, dtype=complex_type)
    
def zeros(m: int, n: int):
    r"""Constructs a $m \times n$ zero matrix."""
    return np.zeros(shape=(m, n), dtype=complex_type)

def eye(n: int):
    r"""Constructs the $n \times n$ identity matrix."""
    return np.eye(n, dtype=complex_type)