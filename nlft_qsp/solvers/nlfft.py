
from .. import numerics as bd

from ..nlft import NonLinearFourierSequence
from ..poly import Polynomial

def nlfft_recurse(a_star: Polynomial, b: Polynomial) -> tuple[NonLinearFourierSequence, Polynomial, Polynomial]:
    n = a_star.effective_degree() + 1

    if n == 0:
        return NonLinearFourierSequence([0]), Polynomial([0]), Polynomial([1])
    
    if n == 1:
        F0 = b[0]/a_star[0]
        r = bd.sqrt(1 + bd.abs2(F0))
        return NonLinearFourierSequence([F0]), Polynomial([F0/r]), Polynomial([1/r])
    
    m = -(-n//2) # ceil(n/2)

    Fup, xi_m, eta_m = nlfft_recurse(a_star.truncate(0,m-1), b.truncate(0,m-1))

    am_star = eta_m.conjugate() * a_star + xi_m.conjugate() * b
    bm = eta_m * b - xi_m * a_star
    bm.support_start -= m
    # Multiplies by z^(-m) (using the shift() method would duplicate the Polynomial object unnecessarily)

    Fdown, xi_mn, eta_mn = nlfft_recurse(am_star.truncate(0, n-m-1), bm.truncate(0, n-m-1))

    xi_n = eta_m.sharp() * xi_mn + xi_m * eta_mn
    eta_n = eta_m * eta_mn - xi_m.sharp() * xi_mn

    return NonLinearFourierSequence(Fup.coeffs + Fdown.coeffs), xi_n, eta_n

def inlft(a: Polynomial, b: Polynomial):
    """Computes the Inverse Non-Linear Fourier Transform using the Non-Linear Fast Fourier Transform algorithm (arXiv:2505.12615).

    Args:
        a, b (Polynomial): The pair `(a, b)` is the NLFT we want to compute the sequence for.

    Note:
        `a` must be outer. To generate an outer complementary polynomial, you can use `weiss.complete`.

    Returns:
        NonLinearFourierSequence: A sequence whose NLFT is equal to `(a, b)` (up to working precision).
    """
    if len(a.support()) != len(b.support()) or a.support().stop != 1:
        return ValueError("(a, b) must be in the image of the NLFT.")
    
    sup_start = b.support_start
    b = b.shift(-sup_start)

    n = len(a.support())
    F, _, _ = nlfft_recurse(a.conjugate(), b)
    return NonLinearFourierSequence(F.coeffs, support_start=sup_start)