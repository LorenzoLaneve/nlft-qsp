
from . import numerics as bd

from .poly import ComplexL0Sequence, Polynomial


class NonLinearFourierSequence(ComplexL0Sequence):
    """
    Class representing a finitely supported sequence of complex numbers over Z.
    The class provides methods to compute the Non-Linear Fourier Transform (NLFT) associated with the sequence.
    """

    def __init__(self, coeffs=[], support_start=0):
        """
        Initializes a non-linear Fourier sequence with a given list of complex values and support starting index.

        Args:
            coeffs (list of complex): A list of complex numbers representing the sequence. The list includes both the lower 
                                      and upper bounds of the sequence.
            support_start (int): The index of the first element of the sequence in Z. The support of the sequence will 
                                 be in the range [support_start, support_start + len(coeffs)].
        """
        super().__init__(coeffs, support_start)
    
    def transform_bounds(self, inf, sup) -> tuple[Polynomial, Polynomial]:
        """
        Computes the Non-Linear Fourier Transform SU(2) for the subsequence within the specified range.

        Args:
            inf (int): The lower bound (included) index of the sequence for the transformation.
            sup (int): The upper bound (excluded) index of the sequence for the transformation.

        Returns:
            tuple[Polynomial, Polynomial]: The SU(2)-NLFT of the subsequence in [inf, sup].
        """
        if sup - inf <= 0:
            return Polynomial([bd.make_complex(1)]), Polynomial([bd.make_complex(0)])

        if sup - inf <= 1:
            F = bd.make_complex(self[inf])
            den = bd.sqrt(1 + bd.abs2(F))
            return Polynomial([1/den]), Polynomial([F/den], inf)  # (1/den, F/den z^inf)
        
        mid = (sup + inf) // 2
        a1, b1 = self.transform_bounds(inf, mid)
        a2, b2 = self.transform_bounds(mid, sup)

        return a1 * a2 - b1 * b2.conjugate(), a1 * b2 + b1 * a2.conjugate()

    def transform(self) -> tuple[Polynomial, Polynomial]:
        """
        Computes the Non-Linear Fourier Transform (NLFT) over SU(2) associated with this sequence.

        Returns:
            tuple[Polynomial, Polynomial]: The SU(2)-NLFT of the sequence.
        """
        n = len(self.coeffs) - 1
        a, b = self.transform_bounds(self.support_start, self.support_start + n + 1)

        if n < 0:
            return a, b

        return a.truncate(-n, 0), b.truncate(self.support_start, self.support_start + n)


