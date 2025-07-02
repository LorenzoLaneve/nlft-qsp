
import numerics as bd

from nlft import NonLinearFourierSequence
from poly import Polynomial


def toeplitz(c: Polynomial, k: int):
    r"""Returns the `(n-k+1) x (n-k+1)` Toeplitz matrix constructed with the coefficients of c (`n` is the index of the last coefficient of c). Specifically, the first column of the matrix will be `[c[n], c[n-1], ..., c[k]]`, i.e., the reversed order of its coefficients in [k, n].

    Note:
        Assuming that c is supported on :math:`(-\infty, n]`, this matrix will act as the linear operator :math:`f \mapsto z^k P (c z^{-k} f)`, where :math:`P` is the Cauchy projection annihilating any negative-degree term.

    Args:
        c (Polynomial): The polynomial c whose coefficients will be arranged in the matrix.
    """
    n = c.support().stop - 1

    return bd.matrix([[c[n + i - j] for i in range(0, n-k+1)] for j in range(0, n-k+1)])

def system_matrix(c: Polynomial, k: int):
    r"""Returns a block matrix of the form `[[I; -T.T], [mp.conj(T); I]]`, where `T` is the
    Toeplitz matrix of `c` and and `I` is the identity matrix. Solving the associated
    linear system yields the Riemann-Hilbert factorization of `(a, b)`, whose ratio is
    approximated by `c`. The Toeplitz matrix will be computed using `toeplitz(c, k)`,
    see the documentation for `toeplitz` for more information.

    Args:
        c (Polynomial): The Laurent polynomial `c` that approximates `b/a`. It should have support in :math:`(-\infty, d]`.
        k (int): Parameter passed to the `toeplitz` function. The function will compute the Riemann-Hilbert factorization from the k-th element of the NLFT. This parameter should not go over the end of the support of `c`.
    """
    n = c.support().stop - 1
    d = n - k


    T = toeplitz(c, k)
    M = bd.zeros(2*d+2, 2*d+2)
    for i in range(d+1):
        for j in range(d+1):
            M[i, (d+1)+j] = -T[j, i]
            M[(d+1)+i, j] = bd.conj(T[i, j])

    for i in range(2*d+2):
        M[i, i] = 1

    return M

def factorize(c: Polynomial, k: int, normalize: bool = False):
    """Computes the (right) Riemann-Hilbert factorization of :math:`(a, b)`, i.e., the pair of polynomials :math:`(a_+, b_+)` such that :math:`(a, b) = (a_-, b_-) (a_+, b_+)`, where :math:`b_+` has support in `[k, n]`.
    
    Note:
        The two polynomials are given up to a common multiplicative constant.

    Args:
        c (Polynomial): The polynomial `c` that approximates the ratio `b/a`.
        k (int): The displacement, i.e., the point in the sequence where the cut should be made.
        normalize (bool): whether to normalize the polynomials so that :math:`|Ap|^2 + |Bp|^2 = 1`. Defaults to False.

    Returns:
        (Polynomial, Polynomial): A pair of polynomials `(Ap, Bp)` equal (up to a `mp.sqrt(Ap[0])` factor, if normalize=False) to the right Riemann-Hilbert factorization of `(a, b)`, whose ratio is approximated by `c`.
    """
    n = c.support().stop - 1
    d = n - k

    A = system_matrix(c, k)
    x = bd.solve_system(A, [0] * (2*d+1) + [1])

    Ap = Polynomial(x[d+1:2*d+2], support_start=-d)
    Bp = Polynomial(x[0:d+1])

    if normalize:
        a_inf = bd.sqrt(Ap[0])
        Bp /= a_inf
        Ap /= a_inf
    
    return Ap, Bp

def inlft(b: Polynomial, c: Polynomial) -> NonLinearFourierSequence:
    """Compute the Inverse Non-Linear Fourier Transform using the Riemann-Hilbert algorithm.

    Args:
        b (Polynomial): The starting polynomial, such that `(a, b)` is the NLFT we want to compute the sequence for.
        c (Polynomial): A polynomial approximating the ratio `b/a`. The end of its support must coincide with the one of `b`.

    Returns:
        NonLinearFourierSequence: A sequence whose NLFT is equal to `(a, b)` (up to working precision).
    """
    if c.support().stop != b.support().stop:
        raise ValueError("The supports of `b` and `c` must end at the same point.")

    F = []
    for k in b.support():
        Ap, Bp = factorize(c, k)

        F.append(Bp[0]/Ap[0])

    return NonLinearFourierSequence(F, b.support_start)