
from numbers import Number

import numerics as bd
from numerics import coeffs_pad
from numerics.backend import generic_complex, generic_real

from util import next_power_of_two, sequence_shift


class ComplexL0Sequence:
    """Represents a sequence of complex numbers index by Z, whose support is finite.
    
    Attributes:
        coeffs (list[generic_complex]): List of complex coefficients.
        support_start (int): Index of the first element of the sequence.
    """

    def __init__(self, coeffs: list[generic_complex], support_start: int = 0):
        """Initializes a complex sequence.

        Args:
            coeffs: List of complex numbers as coefficients.
            support_start (optional): Index of the first element of the sequence. Defaults to 0.
        """
        self.coeffs = [bd.make_complex(c) for c in coeffs]
        self.support_start = support_start

    def support(self) -> range:
        """Returns the range in Z where the sequence is non-zero.

        Note:
            This simply checks the allocated array of coefficients but does not check leading or trailing zeros.

        Returns:
            range: The support of the sequence.
        """
        return range(self.support_start, self.support_start + len(self.coeffs))
    
    def __getitem__(self, k: int) -> generic_complex:
        """Returns the k-th element of the sequence, i.e., F_k.

        Args:
            k (int): The index of the sequence.

        Returns:
            complex: The coefficient of F_k, or 0 if k is out of the support.
        """
        if k in self.support():
            return self.coeffs[k - self.support_start]
        return bd.make_complex(0)

    def __setitem__(self, k: int, c: generic_complex):
        """Sets the coefficient of z^k to be c, allocating space if needed.

        Args:
            k (int): The exponent of z.
            c (complex): The coefficient to set.
        """
        if self.support_start + len(self.coeffs) <= k:
            self.coeffs.extend([bd.make_complex(0)] * (k - self.support_start - len(self.coeffs) + 1))
        elif self.support_start > k:
            self.coeffs = [bd.make_complex(0)] * (self.support_start - k) + self.coeffs
            self.support_start = k
        self.coeffs[k - self.support_start] = bd.make_complex(c)

    def l1_norm(self) -> generic_real:
        """Computes the l1 norm of the sequence.

        Returns:
            float: The sum of absolute values of coefficients.
        """
        return sum(bd.abs(c) for c in self.coeffs)

    def l2_norm(self) -> generic_real:
        """Computes the l2 norm.

        Returns:
            float: The l2 norm.
        """
        return bd.sqrt(self.l2_squared_norm())

    def l2_squared_norm(self) -> generic_real:
        """Computes the squared l2 norm.

        Returns:
            float: The squared l2 norm, i.e., the sum of the squared absolute values.
        """
        return sum(bd.abs2(c) for c in self.coeffs)
    
    def is_real(self) -> bool:
        """Whether the sequence has only real elements."""
        return all(bd.im(F) <= bd.machine_threshold() for F in self.coeffs)
    
    def is_imaginary(self) -> bool:
        """Whether the sequence has only imaginary elements."""
        return all(bd.re(F) <= bd.machine_threshold() for F in self.coeffs)
    
    def is_symmetric(self) -> bool:
        """Whether the sequence satisfies F[k] = F[-k]."""
        for k in self.support():
            if abs(self[k] - self[-k]) > bd.machine_threshold():
                return False
        return True
    
    def __add__(self, other):
        if isinstance(other, Number):
            q = self.duplicate()
            q[0] += other

            return q
        elif not isinstance(other, Polynomial):
            raise TypeError("Polynomial addition admits only other polynomials or scalars.")
                
        self_end = self.support_start + len(self.coeffs)
        other_end = other.support_start + len(other.coeffs)
        
        sum_start = min(self.support_start, other.support_start)
        sum_end = max(self_end, other_end)

        sum_coeffs = []
        for k in range(sum_start, sum_end):
            res = bd.make_complex(0)
            
            if self.support_start <= k and k < self_end:
                res += self.coeffs[k - self.support_start]

            if other.support_start <= k and k < other_end:
                res += other.coeffs[k - other.support_start]

            sum_coeffs.append(res)
            
        return Polynomial(sum_coeffs, sum_start)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return Polynomial([-c for c in self.coeffs], self.support_start)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)


class Polynomial(ComplexL0Sequence):
    """Represents a general Laurent polynomial of one complex variable.

    Attributes:
        coeffs (list[generic_complex]): List of complex coefficients.
        support_start (int): Minimum degree that appears in the polynomial.
    """

    def __init__(self, coeffs: list[generic_complex], support_start: int = 0):
        """Initializes a Polynomial instance.

        Args:
            coeffs: List of complex numbers as coefficients.
            support_start (optional): Minimum degree in the polynomial. Defaults to 0.
        """
        super().__init__(coeffs, support_start)

    def duplicate(self):
        """Creates a duplicate of the current polynomial.

        Returns:
            Polynomial: A new Polynomial instance with the same coefficients and support.
        """
        return Polynomial(self.coeffs, self.support_start)
    
    def shift(self, k: int):
        """Creates a new polynomial equal to the current one, multiplied by `z^k`."""
        return Polynomial(self.coeffs, self.support_start + k)

    def effective_degree(self) -> int:
        """Returns the size of the support of the polynomial minus 1 (max degree - min degree).

        Note:
            This does not check for leading or trailing zeros in the coefficient array.

        Returns:
            int: The effective degree of the polynomial.
        """
        return len(self.coeffs) - 1

    def conjugate(self):
        r"""Returns the conjugate polynomial on the unit circle. If :math:`p(z) = \sum_k p_k z^k`, then its conjugate is defined as :math:`p^*(z) = \sum_k p_k^* z^{-k}`

        Returns:
            Polynomial: The conjugate polynomial.
        """
        conj_coeffs = [bd.conj(x) for x in reversed(self.coeffs)]
        return Polynomial(conj_coeffs, -(self.support_start + len(self.coeffs) - 1))
    
    def sharp(self):
        r"""Same as `conjugate()`, but the support_start is left unchanged.

        Returns:
            Polynomial: The sharp-conjugate polynomial.
        """
        p = self.conjugate()
        p.support_start += self.effective_degree() + 1
        return p

    def schwarz_transform(self):
        r"""Returns the anti-analytic polynomial whose real part gives the current polynomial.
        
        In other words, this is equivalent to adding :math:`iH[p]`, where :math:`H[p]` is the Hilbert transform of p.

        Returns:
            Polynomial: The Schwarz transform of the polynomial.
        """
        schwarz_coeffs = []
        for k in self.support():
            if k < 0:
                schwarz_coeffs.append(2*self[k])
            elif k == 0:
                schwarz_coeffs.append(self[k])

        return Polynomial(schwarz_coeffs, self.support_start)

    def __mul__(self, other):
        if isinstance(other, Number):
            return Polynomial([other * c for c in self.coeffs], self.support_start)
        elif not isinstance(other, Polynomial):
            raise TypeError("Polynomial addition admits only other polynomials or scalars.")
        len_c = len(self.coeffs) + len(other.coeffs) - 1

        # TODO use extra precision here
        coeffs_a = bd.fft(coeffs_pad(self.coeffs, next_power_of_two(len_c)))
        coeffs_b = bd.fft(coeffs_pad(other.coeffs, next_power_of_two(len_c)))

        # Multiply in the Fourier domain
        coeffs_c = [a * b for a, b in zip(coeffs_a, coeffs_b)]

        # Inverse FFT to get the result
        new_coeffs = bd.ifft(coeffs_c)
        support_start = self.support_start + other.support_start  # Lowest degree of the new poly

        return Polynomial(new_coeffs[0:len_c], support_start)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Number):
            return Polynomial([c / other for c in self.coeffs], self.support_start)
        
        raise TypeError("Polynomial division is only possible with scalars.")

    def __call__(self, z) -> generic_complex:
        """Evaluates the polynomial using Horner's method.

        Args:
            z (complex): The point at which to evaluate the polynomial.

        Returns:
            complex: The evaluated result.
        """
        res = self.coeffs[-1]
        for k in reversed(range(len(self.coeffs) - 1)):
            res = res * z + self.coeffs[k]
        return res * (z ** self.support_start)

    def eval_at_roots_of_unity(self, N: int) -> list[generic_complex]:
        """Evaluates the polynomial at the N-th roots of unity using the inverse FFT.

        Args:
            N (int): A power of two specifying the number of roots. If N is not a power of two, then the next power of two is taken.

        Returns:
            list[complex]: List of evaluations at the N-th roots of unity.
            The k-th element will be `self[w^k]`, where `w` is the main N-th root of unity.
        """
        N = next_power_of_two(N)
        M = next_power_of_two(max(N, len(self.coeffs)))

        coeffs = coeffs_pad(self.coeffs, M)
        coeffs = sequence_shift(coeffs, self.support_start)
        # This has the effect of having everything multiplied by z^s

        evals = bd.ifft(coeffs, normalize=False) # M evaluations at the M-th roots of unity
        return evals[::M//N]
    
    def sup_norm(self, N=1024):
        """Estimates the supremum norm of the polynomial over the unit circle
        
        Args:
            N (int, optional): the number of samples to compute the maximum from. If N is not a power of two, then the next power of two is taken.

        Returns:
            float: An estimate for the supremum norm of the polynomial over the unit circle.
        """
        return max([abs(sample) for sample in self.eval_at_roots_of_unity(N)])
    
    def truncate(self, m: int, n: int):
        """Keeps only the coefficients in [m, n], discarding the others.

        Args:
            m (int): Lower bound of degree.
            n (int): Upper bound of degree.

        Returns:
            Polynomial: A new, truncated polynomial.
        """
        return Polynomial([self[k] for k in range(m, n+1)], m)
    
    def only_positive_degrees(self):
        """Discards all the negative degrees, keeping only the non-negative ones.
        
        Returns:
            Polynomial: A new polynomial containing only the positive-degree coefficients."""
        return self.truncate(0, self.support_start + len(self.coeffs) - 1)

    def __str__(self):
        """Converts the polynomial to a human-readable string representation.

        Returns:
            str: The string representation of the polynomial.
        """
        return ' + '.join(f"{c} z^{self.support_start + k}" for k, c in enumerate(self.coeffs))
    

class ChebyshevTExpansion(ComplexL0Sequence):
    """Linear combination of Chebyshev polynomials of the first kind.
    
    Args:
        c: Either the coefficients of the linear combination, or the symmetric Laurent polynomial `P(z)` which are equal up to the change of variable `x = (z + z^(-1))/2`.
    """
    def __init__(self, c: list[generic_complex] | Polynomial):
        if isinstance(c, list):
            super().__init__(c, support_start=0)
        elif isinstance(c, Polynomial):
            if not c.is_symmetric():
                raise ValueError("The given Laurent polynomial is not symmetric.")

            coeffs = [2*c[k] for k in range(c.support().stop)]
            coeffs[0] /= 2
            super().__init__(coeffs, support_start=0)
        else:
            raise ValueError("Only a coefficient vector or symmetric Laurent polynomials are allowed.")
        
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def __call__(self, x: generic_real) -> generic_complex:
        """Evaluates the Chebyshev expansion at the given number.

        Args:
            x (real): The point at which to evaluate the expansion.

        Returns:
            complex: The evaluated result.
        """
        theta = bd.arccos(x)

        return sum(self[k] * bd.cos(k * theta) for k in self.support())
    
    def to_laurent(self):
        """Returns the Laurent polynomial `P(z) = self((z + z^(-1))/2)`."""
        P = Polynomial(list(reversed(self.coeffs)) + self.coeffs[1:], support_start=-len(self.coeffs)+1)
        P[0] *= 2
        return P/2
    
    @classmethod
    def from_polynomial(cls, P: Polynomial):
        """Returns the Chebyshev expansion `T` satisfying `T(x) = P(x)`."""
        return ChebyshevTExpansion(bd.poly2cheb(P.coeffs))
    
    @classmethod
    def from_laurent_polynomial(cls, P: Polynomial):
        """Returns the Chebyshev expansion `T` satisfying `T(x) = (P(z) + P^*(z))/2`.
        
        Note: `P` must be symmetric."""
        if not P.is_symmetric():
            raise ValueError("The given Laurent polynomial is not symmetric.")

        coeffs = [2*P[k] for k in range(P.support().stop)]
        coeffs[0] /= 2

        return ChebyshevTExpansion(coeffs)
    
    def to_polynomial(self) -> Polynomial:
        """Returns the polynomial `P` satisfying `P(x) = T(x)`."""
        return Polynomial(bd.cheb2poly(self.coeffs))