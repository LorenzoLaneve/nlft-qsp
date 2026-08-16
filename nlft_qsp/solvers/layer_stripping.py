import numpy as np

from ..nlft import NonLinearFourierSequence
from ..poly import Polynomial

def inlft(a: Polynomial, b: Polynomial) -> NonLinearFourierSequence:
    """Computes the inverse nonlinear Fourier transform using the plain layer stripping algorithm.

    Args:
        a (Polynomial): The pair $(a, b)$ is the NLFT we want to compute the sequence for.
        b (Polynomial): The pair $(a, b)$ is the NLFT we want to compute the sequence for.

    Note:
        This algorithm is guaranteed to be numerically stable only if $a$ is outer.
        To generate an outer complementary polynomial, you can use `weiss.complete`.

    Returns:
        A sequence whose NLFT is equal to $(a, b)$.
    """
    if len(a.support()) != len(b.support()) or a.support().stop != 1:
        return ValueError("(a, b) must be in the image of the NLFT.")
    
    n = a.effective_degree()

    a_star = a.conjugate()
    b = b.duplicate()

    F = []
    for k in range(n+1):
        Fk = b[0]/a_star[0]
        F.append(Fk)

        s = np.sqrt(1 + Fk * np.conj(Fk))
        a_star, b = (a_star + np.conj(Fk) * b).truncate(0, n-k)/s, \
                    (b - Fk * a_star).truncate(b.support_start+1, b.support_start+n-k)/s

        b.support_start -= 1 # divide by z

    return NonLinearFourierSequence(F, support_start=b.support_start)