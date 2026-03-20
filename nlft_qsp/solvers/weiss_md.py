
from .. import numerics as bd

from ..solvers.weiss import WEISS_MAX_ATTEMPTS, WeissConvergenceError
from ..util import next_power_of_two
from ..poly_md import PolynomialMD, deep_inplace, deep_sequence_shift

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