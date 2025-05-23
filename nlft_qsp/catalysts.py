#
# EXPERIMENTAL SOLVER
# Adversary bound based GQSP solver.
#

from matplotlib.pylab import f
from nlft import NonLinearFourierSequence
import numerics as bd

import numpy as np
import scipy as sp

from poly import Polynomial
from qsp import PhaseFactors

def full_inversion(X):
    """X -> P X P^\dag, where P is the inversion matrix"""
    return np.flip(np.flip(X, axis=0), axis=1)

def lower_cholesky(X):
    """Decompose X = L^\dag L, for a lower triangular matrix L."""
    R = sp.linalg.cholesky(full_inversion(X), check_finite=False) # X = P R.H R P^\dag

    return full_inversion(R) # L = P R P^\dag


def solve_adversary_bound(P: Polynomial, Q: Polynomial):
    """Returns the unique feasible solution of the adversary bound for the (P, Q) state generation problem."""

    if (P * P.conjugate() + Q * Q.conjugate() - 1).l2_norm() > bd.machine_threshold():
        raise ValueError("P and Q must be normalized.")
    
    if P.support() != Q.support():
        raise ValueError("P and Q must have same support.")
    
    n = P.effective_degree()

    e = np.zeros((n+1, n+1), dtype=complex)
    for k in range(n+1):
        for h in range(n+1):
            e[k, h] = - np.conj(P[k]) * P[h] - np.conj(Q[k]) * Q[h]
    e[0,0] += 1


    X = np.zeros((n, n), dtype=complex)
    for k in range(n):
        for h in range(n):
            X[k, h] = sum(e[k - j, h - j] for j in range(min(k, h)+1))

    return X

def catalyst_polynomials(X, reverse=False) -> list[Polynomial]:
    """Returns a list of n polynomials v[0], ..., v[n-1], where n is the dimension of X.
    
    Args:
        reverse (bool): whether the transformation `v[k] -> z^k v(z^{-1})` should be applied."""
    n = X.shape[0]
    v = []

    L = lower_cholesky(X) # X = L.H @ L
    for k in range(n):
        if reverse:
            v.append(Polynomial(reversed(list(L[k, :(k+1)]))))
        else:
            v.append(Polynomial(list(L[k, :(k+1)])))

    return v

def catalyst_consumer(P: Polynomial, Q: Polynomial) -> NonLinearFourierSequence:
    """Computes the inverse NLFT for (z^{-n} P, Q) using the catalyst consumer algorithm [EXPERIMENTAL].
    
    Note:
        We assume P has positive leading coefficient, so that `(z^{-n} P, Q)` can be in the image of the NLFT.
        The algorithm might become numerically unstable if the leading coefficient of P is too close to 0."""

    if (P * P.conjugate() + Q * Q.conjugate() - 1).l2_norm() > bd.machine_threshold():
        raise ValueError("The two polynomials must be normalized.")

    if P.support() != Q.support():
        raise ValueError("P and Q must have same support.")
    
    n = P.effective_degree()

    if (bd.abs(bd.im(P[n])) > bd.machine_threshold() or bd.re(P[n]) < 0):
        raise ValueError("The leading coefficient of P must be real and positive.")
    
    if n <= 1:
        return NonLinearFourierSequence([-np.conj(Q[0]/P[0])], support_start=Q.support_start)

    X = solve_adversary_bound(P, Q)
    v = catalyst_polynomials(X)

    r = [bd.sqrt((1/v[0][0]) ** 2 - 1)] + \
        [bd.sqrt((v[k-1][k-1]/v[k][k]) ** 2 - 1) for k in range(1, n)] # r_k = |F_k|
    
    argF = [bd.pi() + np.angle(Q[0]) - np.angle(v[k][0]) for k in range(n)]
    argF[0] = np.angle(Q[0])

    Fn = -bd.conj(P[0]/Q[0])
    return NonLinearFourierSequence([r[k] * bd.exp(1j*argF[k]) for k in range(n)] + [Fn], support_start=Q.support_start)