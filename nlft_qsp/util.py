"""Utility functions."""

import numpy as np

def next_power_of_two(n):
    r"""Returns the smallest power of two that is $\ge n$."""
    return 1 << (n - 1).bit_length()

def unitroots(N: int):
    r"""Returns a list containing the $N$-th roots of unity."""
    return [np.exp(2j*np.pi*k/N) for k in range(N)]