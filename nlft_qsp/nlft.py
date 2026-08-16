
import numpy as np

from .numerics import complex_type
from .poly import ComplexL0Sequence, Polynomial


class NonLinearFourierSequence(ComplexL0Sequence):
    r"""Class representing a finitely supported sequence of complex numbers over $\mathbb{Z}$.
    The class provides methods to compute the nonlinear Fourier transform (NLFT) associated with the sequence.
    """

    def __init__(self, coeffs: list[complex_type] | np.ndarray=[], support_start: int=0):
        r"""Initializes a nonlinear Fourier sequence with a given list of complex values and support starting index.

        Args:
            coeffs (list[complex_type] | np.ndarray): A list of complex numbers representing the sequence. The list includes both the lower 
                                      and upper bounds of the sequence.
            support_start (int): The index of the first element of the sequence in $\mathbb{Z}$. The support of the sequence will 
                                 be in the range [`support_start`, `support_start + coeffs.shape[0]`].
        """
        super().__init__(coeffs, support_start)
    
    def transform_bounds(self, inf, sup) -> tuple[Polynomial, Polynomial]:
        """
        Computes the nonlinear Fourier transform over $SU(2)$ for the subsequence within the specified range.

        Args:
            inf (int): The lower bound (included) index of the sequence for the transformation.
            sup (int): The upper bound (excluded) index of the sequence for the transformation.

        Returns:
            The $SU(2)$-NLFT of the subsequence in [`inf`, `sup`).

        Note:
            This is used only internally in order to compute the polynomials through a divide-and-conquer strategy.
            If only interested in the final polynomials, please refer to `transform()`.
        """
        if sup - inf <= 0:
            return Polynomial([1]), Polynomial([0])

        if sup - inf <= 1:
            F = self[inf]
            den = np.sqrt(1 + F * np.conj(F))
            return Polynomial([1/den]), Polynomial([F/den], inf)  # (1/den, F/den z^inf)
        
        mid = (sup + inf) // 2
        a1, b1 = self.transform_bounds(inf, mid)
        a2, b2 = self.transform_bounds(mid, sup)

        return a1 * a2 - b1 * b2.conjugate(), a1 * b2 + b1 * a2.conjugate()

    def transform(self) -> tuple[Polynomial, Polynomial]:
        r"""Computes the nonlinear Fourier transform $(a(z), b(z))$ over $SU(2)$ associated with this sequence.
                
        $$\mathrm{NLFT}(F) = (a, b)$$

        See [here](https://arxiv.org/abs/2503.03026) for a definition of the nonlinear Fourier transform.

        Returns:
            The $SU(2)$-NLFT of the sequence.
        """
        n = self.coeffs.shape[0] - 1
        a, b = self.transform_bounds(self.support_start, self.support_start + n + 1)

        if n < 0:
            return a, b

        return a.truncate(-n, 0), b.truncate(self.support_start, self.support_start + n)


