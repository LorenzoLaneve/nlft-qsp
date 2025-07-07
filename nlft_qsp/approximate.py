import numpy as np

from poly import ChebyshevTExpansion


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