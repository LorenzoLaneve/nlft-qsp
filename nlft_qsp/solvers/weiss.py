
from math import ceil

import numerics as bd

from poly import Polynomial
from poly_md import PolynomialMD, deep_inplace, deep_sequence_shift
from util import next_power_of_two, sequence_shift

WEISS_MAX_ATTEMPTS = 3

class WeissConvergenceError(Exception):
    """This error is thrown where Weiss' algorithm does not converge, more precisely,
    when the error in Weiss' algorithm does not improve after `WEISS_MAX_ATTEMPTS` steps."""

    def __init__(self, *args):
        super().__init__(*args)

def laurent_approximation(points: list) -> Polynomial:
    r"""Returns a Laurent polynomial passing through the given points.

    Note:
        `N = len(points)` is assumed to be a power of two.

    Args:
        points (list[complex]): list of values, where the k-th element is considered to be :math:`f(e^{2\pi i k/N})`.

    Returns:
        Polynomial: The unique Laurent polynomial `P(z)` of degree `N = len(points)` satisfying :math:`P(e^{2\pi i k/N}) = f(e^{2\pi i k/N})`, up to working precision, whose frequencies are shifted to be in :math:`[-N/2, N/2)`
    """
    N = len(points)

    coeffs = bd.fft(points, normalize=True)
    coeffs = sequence_shift(coeffs, -N//2) # Zero frequency in the middle

    return Polynomial(coeffs, support_start=-N//2)

def weiss_internal(b: Polynomial, eps:float=-1, compute_ratio=False, verbose=False):
    """Internal function for Weiss' algorithm. The user should call `weiss.complete`, or `weiss.ratio`.

    Args:
        b (Polynomial): The starting polynomial to complete.
        eps (float): The desired tolerance. If not specified, it will be set to working precision.
        compute_ratio (bool, optional): If True, then also a Polynomial approximating :mode:`b/a` will be returned. Defaults to False.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        Polynomial: A polynomial :math:`a(z)` satisfying :math:`|a|^2 + |b|^2 = 1` on the unit circle (up to working precision). If `compute_ratio=True`, a second polynomial :math:`c(z)` that approximates :math:`b/a` is returned.
    """
    d = b.effective_degree()
    if eps < 0:
        eps = 100 * bd.machine_eps()

    eta = 1 - b.sup_norm(max(4000, 4*d))

    N = next_power_of_two(int(d/eta))//2 # Exponential search on N
    threshold = 1
    attempts = 0
    while threshold > eps:
        N *= 2
        if verbose:
            print(f"N = {N}")

        b_points = b.eval_at_roots_of_unity(N)

        R = laurent_approximation([bd.log(1 - bd.abs2(bz))/2 for bz in b_points])

        G = R.schwarz_transform()
        G_points = G.eval_at_roots_of_unity(N)

        a = laurent_approximation([bd.exp(gz) for gz in G_points])
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
        c = laurent_approximation([bz * bd.exp(-gz) for bz, gz in zip(b_points, G_points)])
        return a, c.truncate(c.support_start, b.support().stop - 1)
    else:
        return a

def complete(b: Polynomial, eps:float=-1, verbose=False):
    """Uses Weiss' algorithm to find a complementary polynomial to the given one. The polynomial will also be the unique outer, positive-mean polynomial with this property, according to arXiv:2407.05634.

    Args:
        b (Polynomial): The polynomial to complete.
        eps (float): The desired tolerance. If not specified, it will be set to working precision.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        Polynomial: A polynomial :math:`a(z)` satisfying :math:`|a|^2 + |b|^2 = 1` on the unit circle (up to eps).
    """
    return weiss_internal(b, eps, verbose=verbose)

def ratio(b: Polynomial, eps:float=-1, verbose=False):
    """Uses Weiss' algorithm to compute :math:`b/a`, where :math:`a` is the unique outer, positive-mean polynomial such that `|a|^2 + |b|^2 = 1`, up to working precision.

    Args:
        b (Polynomial): The polynomial to complete.
        eps (float): The desired tolerance. If not specified, it will be set to working precision.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        Polynomial: A polynomial :math:`a(z)` satisfying :math:`|a|^2 + |b|^2 = 1` on the unit circle (up to working precision), and a polynomial :math:`c` that approximates :math:`b/a`.
    """
    return weiss_internal(b, eps, compute_ratio=True, verbose=verbose)



def laurent_approximation_md(points: list, m: int) -> PolynomialMD:
    r"""Returns a Laurent polynomial passing through the given points.

    Note:
        `N = len(points)` is assumed to be a power of two.

    Args:
        points (list[complex]): list of values, where the (k_1, ..., k_m)-th element is considered to be :math:`f(e^{2\pi i k_1/N}, ..., e^{2\pi i k_m/N})`.
        m (int): dimension of points, i.e., number of variables for the Laurent approximation.

    Returns:
        PolynomialMD: The unique Laurent polynomial `P(z_1, ..., z_m)` of degree `N = len(points)` satisfying :math:`P(e^{2\pi i k_1/N}, ..., e^{2\pi i k_m/N}) = f(e^{2\pi i k_1/N}, ..., e^{2\pi i k_m/N})`, up to working precision, whose frequencies are shifted to be in :math:`[-N/2, N/2)^m`
    """
    N = len(points)
    support_start = (-N//2,) * m

    coeffs = bd.fft_md(points, normalize=True)
    coeffs = deep_sequence_shift(coeffs, support_start) # Zero frequency in the middle
    
    return PolynomialMD(coeffs, support_start)

def complete_md(b: PolynomialMD, verbose=False):
    """Uses the Weiss-Schwarz algorithm to find a complementary polynomial to the given one. The polynomial will also be the unique outer, positive-mean polynomial with this property.

    Args:
        b (PolynomialMD): The polynomial to complete.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        PolynomialMD: A polynomial :math:`a(z)` satisfying :math:`|a|^2 + |b|^2 = 1` on the unit circle (up to working precision).
    """
    N = 4*next_power_of_two(max(k for k in b.effective_degree())) # Exponential search on N, this can be better.
    m = b.dim # Number of variables

    threshold = 1
    attempts = 0
    while threshold > 100 * bd.machine_eps():
        N *= 2

        b_points = b.eval_at_roots_of_unity(N)
        deep_inplace(b_points, lambda bz: bd.log(1 - bd.abs2(bz))/2)
        R = laurent_approximation_md(b_points, m)

        G = R.schwarz_transform()

        G_points = G.eval_at_roots_of_unity(N)
        deep_inplace(G_points, lambda gz: bd.exp(gz))
        a = laurent_approximation_md(G_points, m)

        #a = a.truncate(-b.effective_degree(), 0) # a and b must have the same support

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

    return a