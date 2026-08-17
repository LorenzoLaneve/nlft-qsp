"""Module for polynomial completion based on Prony's method (taken from [github.com/quantum-programming/gqsp-angle-finding](https://github.com/quantum-programming/gqsp-angle-finding))."""


import numpy as np
from ...poly import Polynomial

from scipy.linalg import hankel
from numpy.linalg import svd


def complete(b: Polynomial) -> Polynomial:
    """Uses Prony's method (arXiv:2202.02671) to find a complementary polynomial to the given one.

    Args:
        b (Polynomial): The polynomial to complete.

    Note:
        Numerical stability is not guaranteed.

    Returns:
        Polynomial: A polynomial $a(z)$ satisfying $|a(z)|^2 + |b(z)|^2 = 1$ on the unit circle.
        In particular $(a, b)$ will be in the image of the NLFT.
    """
    n = b.effective_degree()
    fft_size = 1 << (n.bit_length() + 5)

    F_w = np.array(b.eval_at_roots_of_unity(fft_size))
    completion_poly_inv_w = 1 / (1 - F_w * np.conj(F_w))

    completion_poly_inv_fft = np.flip(np.fft.fft(completion_poly_inv_w))
    c = completion_poly_inv_fft[: n + 2]
    r = completion_poly_inv_fft[n + 1 : n + n + 2]
    M = hankel(c, r)
    _, _, vh = svd(M)

    a_coeffs = np.conj(vh[-1])
    a = Polynomial(a_coeffs)
    C = np.sqrt(np.mean(np.array((1 - b * b.conjugate()).eval_at_roots_of_unity(1024)) / \
                        np.array((a * a.conjugate()).eval_at_roots_of_unity(1024))))
    a *= C
    a = a.shift(-n)
    a *= np.exp(-1j*np.angle(a[0])) # we want a[0] > 0
    return a