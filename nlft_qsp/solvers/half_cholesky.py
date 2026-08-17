import numpy as np
import scipy as sp

from .. import numerics as bd

from ..nlft import NonLinearFourierSequence
from ..poly import Polynomial

from ..numerics import complex_type


def solve_ldl_system(L: np.ndarray, D: np.ndarray, C: np.ndarray, mode: str='toeplitz') -> np.ndarray:
    """Solves the matrix system $LDL^* X = C$.
    
    Args:
        mode (str): if `'hankel'` then the input and output vectors are flipped along axis 0.
    
    Note:
        The following shapes are assumed:
        - `L.shape = (n * d, n * d)`
        - `D.shape = (n, d, d)`, it should NOT be given as a full block-diagonal matrix.
        - `C`.shape = (n, d, *)` or `(n * d, *)`
        The output $X$ will always be of shape `(n * d, *)`."""
    n = D.shape[0]
    d = D.shape[1]

    if mode == 'hankel':
        C = C.reshape(n, d, -1)
        C = np.flip(C, axis=0)

    C = C.reshape(n * d, -1)

    
    Y = sp.linalg.solve_triangular(L, C, lower=True, unit_diagonal=True, check_finite=False)

    # DZ = Y block-wise inverse
    Y = Y.reshape(n, d, -1) # blockify
    Z = np.linalg.solve(D, Y)
    Z = Z.reshape(n * d, -1) # re-flatten
    
    # L^* X = Z backward substitution
    X = sp.linalg.solve_triangular(L, Z, lower=True, trans='C', unit_diagonal=True, check_finite=False)
    if mode == 'hankel':
        X = np.flip(X.reshape(n, d, -1), axis=0).reshape(n * d, -1)
    
    return X
    

def half_cholesky_matrix_ldl(p: np.ndarray) -> np.ndarray:
    r"""Computes the lower triangular block-matrix $L$ and the diagonal positive definite block-matrix $D$ for $I + B B^* = LDL^*$.
    Here $B$ is assumed to be a lower triangular block-Toeplitz matrix whose first block-column is $p$.

    Args:
        p (np.ndarray): The first block column of B. It should be of shape `(n, d1, d2)` where B is $n \times n$ with `(d1, d2)` blocks.

    Returns:
        The matrix $L$ as a numpy array of shape `(n * d1, n * d1)` and $D$ as a list of blocks of shape `(n, d1, d1)`.
        
    Note:
        The $k$-th column of $L$ returned will be of length $(n+1) - k$, meaning that the zeros above the diagonal will not be added.
        $D$ is returned as a list of blocks."""
    N = p.shape[0]
    d1 = p.shape[1]
    d2 = p.shape[2]

    G = np.zeros((N * d1, d1 + d2), dtype=complex_type)
    G[:d1, :d1] = np.eye(d1, dtype=complex_type)
    G[:, d1:] = p.reshape(N * d1, d2)
    # e0 = np.array([np.eye(d1, d1)] + [np.zeros((d1, d1))] * n, dtype=complex_type)
    # G = np.array([np.hstack([uk, vk]) for uk, vk in zip(e0, p)], dtype=complex_type)
    # G = G.reshape(N * d1, d1 + d2) # U is d1 x d1, V is d1 x d2

    L_cols = []
    D_blocks = []
    for k in range(N):
        _, R = sp.linalg.qr(G.conj().T)
        Lk = R.conj().T #Lk @ Q.H = G

        up = Lk[:(N-k) * d1, :d1].reshape(N-k, d1, -1) # [Lk[j * d1 : (j+1) * d1, :d1] for j in range(N-k)]
        vp = Lk[:(N-k) * d1, d1:].reshape(N-k, d1, -1) # [Lk[j * d1 : (j+1) * d1, d1:] for j in range(N-k)]

        L_cols.append(up @ np.linalg.inv(up[0])) # [upj @ np.linalg.inv(up[0]) for upj in up])
        D_blocks.append(up[0] @ up[0].conj().T)

        if k < N - 1:
            G = np.zeros(((N-k-1) * d1, d1 + d2), dtype=complex_type)
            G[:, :d1] = up[:-1].reshape((N-k-1) * d1, -1)
            G[:, d1:] = vp[1:].reshape((N-k-1) * d1, -1)

    L = np.zeros((N * d1, N * d1), dtype=complex_type)
    for k, l in enumerate(L_cols):
        for j, c in enumerate(l):
            L[(k+j) * d1 : (k+j+1) * d1, k * d1 : (k+1) * d1] = c

    return L, np.array(D_blocks, dtype=complex_type)

def solve_system_displacement_structure(p: np.ndarray, C: np.ndarray, mode: str = 'toeplitz') -> np.ndarray:
    """Solves the linear system $(I + BB^*) X = C$, where $p$ is the first column of $B$.
    $B$ will be assumed to be a lower-triangular block-Toeplitz matrix (when `mode='toeplitz'`)
    or a anti-upper triangular block-Hankel matrix (when `mode='hankel'`).

    Args:
        mode (str): whether $B$ is constructed as a `'toeplitz'` or as a `'hankel'` matrix.
    
    Note:
        $p$ and $C$ are assumed to be of shape `(n, d1, d2)` and `(n, d1, *)` respectively."""

    if mode == 'hankel':
        C = np.flip(C, axis=0)
        p = np.flip(p, axis=0)

    N = p.shape[0]
    d1 = p.shape[1]

    L, D = half_cholesky_matrix_ldl(p)

    # LDL^* X = C
    X = solve_ldl_system(L, D, C).reshape(N, d1, -1)

    if mode == 'hankel':
        X = np.flip(X, axis=0)

    return X


def half_cholesky_ldl(u, v) -> np.ndarray:
    r"""Computes the lower triangular matrix $L$ for $U U^* + V V^* = LDL^*$ using the Half-Cholesky method (see [arXiv:2410.06409](https://arxiv.org/abs/2410.06409)).
    Here $D$ is some positive diagonal matrix, while $U, V$ are the Toeplitz matrices containing $u, v$ as first columns, respectively.
    
    Note:
        The $k$-th column will be of length $(n+1)-k$, meaning that the zeros above the diagonal will not be added.

    Returns:
        The matrix $L$, given as an object of the backend."""
    n = len(u) - 1

    G = bd.matrix([[uk, vk] for uk, vk in zip(u, v)])

    L_cols = []
    for k in range(n):
        _, R = sp.linalg.qr(G.conj().T)
        Lk = R.conj().T #Lk @ Q.H = G

        up = [Lk[j, 0] for j in range(n+1-k)]
        vp = [Lk[j, 1] for j in range(n+1-k)]

        L_cols.append([upj/up[0] for upj in up])

        G = bd.matrix([[uk, vk] for uk, vk in zip(up[:-1], vp[1:])])

    L_cols += [[1]] # last column

    L = bd.zeros(n+1, n+1)
    for k, l in enumerate(L_cols):
        for j, c in enumerate(l):
            L[k+j, k] = c

    return L

def inlft(b: Polynomial, c: Polynomial) -> NonLinearFourierSequence:
    """Compute the inverse nonlinear Fourier transform using the Half Cholesky algorithm.

    Args:
        b (Polynomial): The starting polynomial, such that $(a, b)$ is the NLFT we want to compute the sequence for.
        c (Polynomial): A polynomial approximating the ratio $b/a$. The end of its support must coincide with the one of $b$.

    Returns:
        A sequence whose NLFT is equal to $(a, b)$ (up to working precision).
    """
    n = b.effective_degree()

    p = [np.conj(c[k]) for k in reversed(b.support())]

    L = half_cholesky_ldl([1] + [0] * n, p) # (e_0, p)

    F = [0] * (n+1)
    for k in range(n+1): # (F_n^*, ..., F_0^*) = L^{-1} p by Forward substitution
        F[k] = p[k] - sum(L[k, j]*F[j] for j in range(k))

    return NonLinearFourierSequence([np.conj(f) for f in reversed(F)], b.support_start)