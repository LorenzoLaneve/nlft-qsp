
from numbers import Number
from .nlft import NonLinearFourierSequence
from . import numerics as bd

from .poly_md import ComplexL0SequenceMD, PolynomialMD, to_poly_md

# Example:
# [[a, b], [c], [d, e, f], [g]]
#   fg
#   e
# bcd
# a

class StairlikeSequence2D:
    """
    Class representing a compactly supported stairlike sequence in the Z^2 grid.
    The class provides methods to compute the Non-Linear Fourier Transform (NLFT) associated with the sequence.
    """
    def __init__(self, coeffs: list, support_start: tuple[int]=(0,0)):
        if len(support_start) != 2:
            raise ValueError("support_start must be two-dimensional.")

        if not isinstance(coeffs, list):
            raise ValueError("Coefficient list must be of type list.")
        
        self.support_start = support_start

        self.coeffs = []
        self.m = [self.support_start[1]]
        for col in coeffs:
            if not all(isinstance(c, Number) for c in col):
                raise ValueError("coeffs must be a two-dimensional sequence of numbers.")
            
            if len(col) == 0:
                col = [0]
            
            self.coeffs.append([bd.make_complex(c) for c in col])
            self.m.append(self.m[-1] + len(col) - 1)

    def support_x(self):
        return range(self.support_start[0], self.support_start[0] + len(self.coeffs))
    
    def support_y(self):
        return range(self.m[0], self.m[-1] + 1)
    
    def __getitem__(self, k):
        if not (isinstance(k, tuple) and len(k) == 2):
            raise ValueError("Element index must be given as two elements.")
        
        x, y = k
        if x in self.support_x() and (self.m[x] <= y and y <= self.m[x+1]):
            return self.coeffs[x][y - self.m[x]]
        
        return bd.make_complex(0)

    def transform_bounds(self, inf, sup) -> tuple[PolynomialMD, PolynomialMD]:
        """
        Computes the Non-Linear Fourier Transform SU(2) for the subsequence within the specified range.

        Args:
            inf (int): The lower bound (included) index of the sequence for the transformation.
            sup (int): The upper bound (excluded) index of the sequence for the transformation.

        Returns:
            tuple[Polynomial, Polynomial]: The SU(2)-NLFT of the subsequence in [inf, sup].
        """
        if sup - inf <= 0:
            return PolynomialMD([[bd.make_complex(1)]], support_start=(0,0)), PolynomialMD([bd.make_complex(0)], support_start=(0,0))

        if sup - inf <= 1:
            k = inf - self.support_start[0]
            unlft = NonLinearFourierSequence(self.coeffs[k], support_start=self.m[k])
            Gk, Fk = unlft.transform()

            return to_poly_md(Gk, 2), to_poly_md(Fk, 2).shift(inf, 0) # (G_k, F_k z^k)
        
        mid = (sup + inf) // 2
        a1, b1 = self.transform_bounds(inf, mid)
        a2, b2 = self.transform_bounds(mid, sup)

        return a1 * a2 - b1 * b2.conjugate(), a1 * b2 + b1 * a2.conjugate()

    def transform(self) -> tuple[PolynomialMD, PolynomialMD]:
        """
        Computes the Non-Linear Fourier Transform (NLFT) over SU(2) associated with this sequence.

        Returns:
            tuple[Polynomial, Polynomial]: The SU(2)-NLFT of the sequence.
        """
        n = len(self.coeffs) - 1
        a, b = self.transform_bounds(self.support_start[0], self.support_start[0] + len(self.coeffs))

        if n < 0:
            return a, b
        
        b_sup_x = self.support_x()
        b_sup_y = self.support_y()
        
        b_sup = (b_sup_x, b_sup_y)
        a_sup = (range(b_sup_x.start - b_sup_x.stop, 0), range(b_sup_y.start - b_sup_y.stop, 0)) # 1s excluded

        return a.truncate(a_sup), b.truncate(b_sup)


