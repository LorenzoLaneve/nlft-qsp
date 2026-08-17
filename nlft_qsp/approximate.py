"""Provides functions to compute Laurent/Chebyshev/Fourier approximations."""

from typing import Callable

import numpy as np

from .poly import ChebyshevTExpansion, Polynomial
from .util import next_power_of_two

from .numerics import complex_type


def chebyshev_approximate(f: Callable, N: int) -> ChebyshevTExpansion:
    r"""Computes the Chebyshev expansion up to $N$ for a complex-valued function $f : [-1, 1] \rightarrow \mathbb{C}$.

    Args:
        f (Callable): complex-valued function f(x)
        N (int): degree of Chebyshev approximation
    """
    x = np.cos(np.pi * np.arange(N + 1) / N)
    fx = [f(xk) for xk in x]

    fx_m = fx + fx[1:-1][::-1]

    F = np.fft.fft(fx_m)
    F = F[:N + 1] / N
    F[0]  /= 2
    F[-1] /= 2

    return ChebyshevTExpansion(F)

def fourier_approximate(f: Callable, N: int) -> Polynomial:
    r"""Computes the Fourier series of the given function $f(z)$, $z = e^{i\theta}$ being a complex number of unit modulus.

    Args:
        f (Callable): a function taking a complex number $z$ and returning a complex number.
        N (int): The degree of approximation.

    Returns:
        Polynomial: A Laurent polynomial of degrees in $\{-N, N+1, \ldots, N-1\}$ approximating $f$
    """
    M = next_power_of_two(2*N+1)

    zk = np.exp(2j * np.pi * np.arange(M) / M)
    fk = np.array([f(z) for z in zk], dtype=complex)

    pk = (np.fft.fft(fk) / M).tolist()
    
    return Polynomial(pk[M-N:] + pk[:N+1], support_start=-N)

def laurent_approximation(points: list[complex_type | np.ndarray]) -> Polynomial:
    r"""Returns a Laurent polynomial passing through the given points.

    Args:
        points (list[complex_type] | np.ndarray): list of values, where the $k$-th element is considered to be $f(e^{2\pi i k/N})$. If the points are matrices, then the returned approximation will be a matrix polynomial.

    Returns:
        Polynomial: The unique Laurent polynomial $P(z)$ of degree $N = len(points)$ satisfying $P(e^{2\pi i k/N}) = f(e^{2\pi i k/N})$, up to working precision, whose frequencies are shifted to be in $\{-N/2, -N/2+1, \ldots, N/2 - 1\}$.
    """
    N = len(points)

    coeffs = np.fft.fft(points, norm='forward', axis=0)
    coeffs = np.roll(coeffs, -N//2, axis=0) # Zero frequency in the middle

    return Polynomial(coeffs, support_start=-N//2)