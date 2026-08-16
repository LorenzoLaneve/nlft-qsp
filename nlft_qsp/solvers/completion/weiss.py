import numpy as np

from ... import numerics as bd

from ...poly import Polynomial
from ...util import next_power_of_two
from ...approximate import laurent_approximation

WEISS_MAX_ATTEMPTS = 3

class WeissConvergenceError(Exception):
    """This error is thrown where Weiss' algorithm does not converge, more precisely,
    when the error in Weiss' algorithm does not improve after `WEISS_MAX_ATTEMPTS` steps."""

    def __init__(self, *args):
        super().__init__(*args)

def weiss_internal(b: Polynomial, eps:float=-1, compute_ratio=False, verbose=False) -> Polynomial | tuple[Polynomial, Polynomial]:
    """Internal function for Weiss' algorithm. The user should call `weiss.complete`, or `weiss.ratio`.

    Args:
        b (Polynomial): The starting polynomial to complete.
        eps (float): The desired tolerance. If not specified, it will be set to working precision.
        compute_ratio (bool, optional): If True, then also a `Polynomial` approximating $b/a$ will be returned.
        verbose (bool, optional): verbosity during the procedure.

    Returns:
        Polynomial: A polynomial $a(z)$ satisfying $|a|^2 + |b|^2 = 1$ on the unit circle (up to working precision). If `compute_ratio=True`, a second polynomial $c(z)$ that approximates $b/a$ is returned.
    """
    d = b.effective_degree()
    if eps < 0:
        eps = bd.machine_threshold() * 10e-3

    eta = 1 - b.sup_norm(max(4000, 4*d))

    N = next_power_of_two(int(d/eta))//2 # Exponential search on N
    threshold = 1
    attempts = 0
    while threshold > eps:
        N *= 2
        if verbose:
            print(f"N = {N}")

        b_points = b.eval_at_roots_of_unity(N)

        R = laurent_approximation([np.log(1 - bz * np.conj(bz))/2 for bz in b_points])

        G = R.schwarz_transform()
        G_points = G.eval_at_roots_of_unity(N)

        a = laurent_approximation([np.exp(gz) for gz in G_points])
        a = a.truncate(-b.effective_degree(), 0) # a and b must have the same support

        new_thr = (a * a.conjugate() + b * b.conjugate() - 1).l2_norm()
        if verbose:
            print(f"N = {N:>7}, threshold = {new_thr}")

        if threshold <= new_thr:
            attempts += 1
            if attempts >= WEISS_MAX_ATTEMPTS:
                raise WeissConvergenceError()
        else:
            threshold = new_thr
            attempts = 0

    if compute_ratio:
        c = laurent_approximation([bz * np.exp(-gz) for bz, gz in zip(b_points, G_points)])
        return a, c.truncate(c.support_start, b.support().stop - 1)
    else:
        return a

def complete(b: Polynomial, eps:float=-1, verbose=False) -> Polynomial:
    """Uses Weiss' algorithm to find a complementary polynomial to the given one. The polynomial will also be the unique outer, positive-mean polynomial with this property, according to [arXiv:2407.05634](https://arxiv.org/abs/2407.05634).

    Args:
        b (Polynomial): The polynomial to complete.
        eps (float): The desired tolerance. If not specified, it will be set to working precision.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        A polynomial $a(z)$ satisfying $|a|^2 + |b|^2 = 1$ on the unit circle (up to eps).
    """
    return weiss_internal(b, eps, verbose=verbose)

def ratio(b: Polynomial, eps:float=-1, verbose=False) -> Polynomial:
    """Uses Weiss' algorithm to compute $b/a$, where $a$ is the unique outer, positive-mean polynomial such that $|a|^2 + |b|^2 = 1$, up to working precision.

    Args:
        b (Polynomial): The polynomial to complete.
        eps (float): The desired tolerance. If not specified, it will be set to working precision.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        A polynomial $a(z)$ satisfying $|a(z)|^2 + |b(z)|^2 = 1$ on the unit circle (up to working precision), and a polynomial $c$ that approximates $b/a$.
    """
    return weiss_internal(b, eps, compute_ratio=True, verbose=verbose)
