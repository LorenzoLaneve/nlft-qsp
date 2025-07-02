
import numerics as bd

from nlft import NonLinearFourierSequence
from poly import Polynomial


def half_cholesky_ldl(u, v):
    r"""Computes the lower triangular matrix `L` for :math:`I + B B^\dag = LDL^\dag` where `D` is some positive diagonal matrix,
    and `B` is the Toeplitz matrix containing `(c^*[n], c^*[n-1], ..., c^*[k])` using the Half-Cholesky method (see arXiv:2410.06409).
    
    Note:
        The k-th column will be of length `(n+1)-k`, meaning that the zeros above the diagonal will not be added.

    Returns:
        list: The matrix L, given as an object of the backend."""
    n = len(u) - 1

    G = bd.matrix([[uk, vk] for uk, vk in zip(u, v)])

    L_cols = []
    for k in range(n):
        _, R = bd.qr_decomp(bd.conj_transpose(G))
        Lk = bd.conj_transpose(R) #Lk @ Q.H = G

        up = [Lk[j, 0] for j in range(n+1-k)]
        vp = [Lk[j, 1] for j in range(n+1-k)]

        L_cols.append([upj/up[0] for upj in up])

        G = bd.matrix([[uk, vk] for uk, vk in zip(up[:-1], vp[1:])])

    L_cols += [[bd.make_complex(1)]] # last column

    L = bd.zeros(n+1, n+1)
    for k, l in enumerate(L_cols):
        for j, c in enumerate(l):
            L[k+j, k] = c

    return L

def inlft(b: Polynomial, c: Polynomial) -> NonLinearFourierSequence:
    """Compute the Inverse Non-Linear Fourier Transform using the Half Cholesky algorithm.

    Args:
        b (Polynomial): The starting polynomial, such that `(a, b)` is the NLFT we want to compute the sequence for.
        c (Polynomial): A polynomial approximating the ratio `b/a`. The end of its support must coincide with the one of `b`.

    Returns:
        NonLinearFourierSequence: A sequence whose NLFT is equal to `(a, b)` (up to working precision).
    """
    n = b.effective_degree()

    p = [bd.conj(c[k]) for k in reversed(b.support())]

    L = half_cholesky_ldl([bd.make_complex(1)] + [bd.make_complex(0)] * n, p) # (e_0, p)

    F = [0] * (n+1)
    for k in range(n+1): # (F_n^*, ..., F_0^*) = L^{-1} p by Forward substitution
        F[k] = p[k] - sum(L[k, j]*F[j] for j in range(k))

    return NonLinearFourierSequence([bd.conj(f) for f in reversed(F)], b.support_start)