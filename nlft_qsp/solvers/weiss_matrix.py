import numpy as np
from sympy import Matrix

from .. import numerics as bd

from ..poly import Polynomial
from ..poly_matrix import MatrixPolynomial
from ..util import next_power_of_two, sequence_shift


def laurent_cholesky_approximation(B: MatrixPolynomial, N: int) -> MatrixPolynomial:
    r"""Returns a Laurent matrix polynomial passing through the given points via Cholesky decomposition.

    Note:
        `N` is assumed to be a power of two.

    Args:
        B (MatrixPolynomial): A matrix polynomial whose values at N points on the unit circle will be used.
        N (int): Number of points on the unit circle to sample.

    Returns:
        MatrixPolynomial: The unique Laurent matrix polynomial `R(z)` of degree `N = len(points)` satisfying :math:`R(e^{2\pi i k/N}) = L_k`, where `L_k` is the lower-triangular Cholesky factor of `B(e^{2\pi i k/N})`, up to working precision, whose frequencies are shifted to be in :math:`[-N/2, N/2)`
    """
    nrows, ncols = B.shape
    R = MatrixPolynomial(B.shape)

    # Sample B at N points on the unit circle and compute Cholesky factors
    cholesky_pointwise = [np.linalg.cholesky(B(bd.exp_2pi_i_k_over_N(k, N))) for k in range(N)]

    # For each matrix entry (i,j), compute inverse DFT of samples to get length-N coeffs
    for i in range(nrows):
        for j in range(ncols):
            samples = np.array([cholesky_pointwise[k][i, j] for k in range(N)], dtype=complex)
            coeffs = np.fft.ifft(samples)  # length N, corresponds to exponents 0..N-1

            # interpret coeffs as Laurent coefficients with support_start = -N//2
            coeffs_list = [bd.make_complex(c) for c in coeffs.tolist()]
            coeffs_list = sequence_shift(coeffs_list, -N//2)  # Zero frequency in the middle
            R[i, j] = Polynomial(coeffs_list, support_start=-N//2)

    return R

def complete_matrix(B: MatrixPolynomial, eps:float=-1, verbose=False):
    """Uses Weiss' algorithm to find a complementary polynomial to the given one. The polynomial will also be the unique outer, positive-mean polynomial with this property, according to arXiv:2407.05634.

    Args:
        B (MatrixPolynomial): The matrix polynomial to complete.
        eps (float): The desired tolerance. If not specified, it will be set to working precision.
        verbose (bool, optional): verbosity during the procedure. Defaults to False.

    Returns:
        MatrixPolynomial: A matrix polynomial :math:`A(z)` satisfying :math:`A^* A + B^* B = 1` on the unit circle (up to eps).
    """
    d = B.effective_degree()
    if eps < 0:
        eps = 100 * bd.machine_eps()

    # choose an FFT-friendly sample size
    N = next_power_of_two(2 * d)

    # Build a Laurent matrix polynomial R(z) of support [-d, d-1] that interpolates
    # the sampled lower-triangular matrices: R(exp(2πik/N)) = cholesky_pointwise[k]
    R = laurent_cholesky_approximation(B, N)

    # R now satisfies R(exp(2πik/N)) == cholesky_pointwise[k] (within numerical error)
    # continue with Weiss algorithm using R...
    ...
