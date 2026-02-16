import numpy as np

from .poly import ChebyshevTExpansion, Polynomial
from .util import next_power_of_two


def chebyshev_approximate(f, N) -> ChebyshevTExpansion:
    """
    Computes the Chebyshev expansion up to `N` for a complex-valued function f on [-1, 1].

    Args:
        f (callable): complex-valued function f(x)
        N (int): degree of Chebyshev approximation
    """
    x = np.cos(np.pi * np.arange(N + 1) / N)
    fx = [f(xk) for xk in x]

    # Correct mirrored extension
    fx_m = fx + fx[1:-1][::-1]

    # FFT without normalization
    F = np.fft.fft(fx_m)
    F = F[:N + 1] / N
    F[0]  /= 2
    F[-1] /= 2

    return ChebyshevTExpansion(F.tolist())

def fourier_approximate(f, N) -> Polynomial:
    r"""Computes the Fourier series of the given function `f(z)`

    Args:
        f (callable): a function taking a complex number z and returning a complex number.
        N (int): The degree of approximation.

    Returns:
        Polynomial: A Laurent polynomial of degrees in `[-N, N)` approximating f
    """
    M = next_power_of_two(2*N+1)

    zk = np.exp(2j * np.pi * np.arange(M) / M)
    fk = np.array([f(z) for z in zk], dtype=complex)

    pk = (np.fft.fft(fk) / M).tolist()
    
    return Polynomial(pk[M-N:] + pk[:N+1], support_start=-N)