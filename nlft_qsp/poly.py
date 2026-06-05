import numpy as np

from numbers import Number

from . import numerics as bd
from .numerics import complex_type, float_type

from .util import coeffs_pad, next_power_of_two, sequence_shift


class ComplexL0Sequence:
    """Represents a sequence of complex numbers index by Z, whose support is finite.
    
    Attributes:
        coeffs (list[complex_type]): List of complex coefficients.
        support_start (int): Index of the first element of the sequence.
    """

    def __init__(self, coeffs: list[complex_type], support_start: int = 0):
        """Initializes a complex sequence.

        Args:
            coeffs: List of complex numbers as coefficients.
            support_start (optional): Index of the first element of the sequence. Defaults to 0.
        """
        if any(not isinstance(ck, Number) for ck in coeffs):
            raise ValueError(f"The list of coefficients must be a single list or np.ndarray of complex values. Got {coeffs}.")

        self.coeffs = bd.matrix(coeffs)
        self.support_start = support_start

    def support(self) -> range:
        """Returns the range in Z where the sequence is non-zero.

        Note:
            This simply checks the allocated array of coefficients but does not check leading or trailing zeros.

        Returns:
            range: The support of the sequence.
        """
        return range(self.support_start, self.support_start + self.coeffs.shape[0])
    
    def __getitem__(self, k: int) -> complex_type:
        """Returns the k-th element of the sequence, i.e., F_k.

        Args:
            k (int): The index of the sequence.

        Returns:
            complex: The coefficient of F_k, or 0 if k is out of the support.
        """
        if k in self.support():
            return self.coeffs[k - self.support_start]
        return 0

    def __setitem__(self, k: int, c: complex_type):
        """Sets the coefficient of z^k to be c, allocating space if needed.

        Args:
            k (int): The exponent of z.
            c (complex): The coefficient to set.
        """
        if self.support_start + self.coeffs.shape[0] <= k:
            self.coeffs = np.pad(self.coeffs, (0, k - self.support_start - self.coeffs.shape[0] + 1))
        elif self.support_start > k:
            self.coeffs = np.pad(self.coeffs, (self.support_start - k, 0))
            self.support_start = k
        self.coeffs[k - self.support_start] = c

    def l1_norm(self) -> float_type:
        """Computes the l1 norm of the sequence.

        Returns:
            float: The sum of absolute values of coefficients.
        """
        return sum(np.abs(c) for c in self.coeffs)

    def l2_norm(self) -> float_type:
        """Computes the l2 norm.

        Returns:
            float: The l2 norm.
        """
        return np.sqrt(self.l2_squared_norm())

    def l2_squared_norm(self) -> float_type:
        """Computes the squared l2 norm.

        Returns:
            float: The squared l2 norm, i.e., the sum of the squared absolute values.
        """
        return sum(c * np.conj(c) for c in self.coeffs)
    
    def is_real(self) -> bool:
        """Whether the sequence has only real elements."""
        return all(np.abs(np.imag(F)) <= bd.machine_threshold() for F in self.coeffs)
    
    def is_imaginary(self) -> bool:
        """Whether the sequence has only imaginary elements."""
        return all(np.abs(np.real(F)) <= bd.machine_threshold() for F in self.coeffs)
    
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
                
        self_end = self.support_start + self.coeffs.shape[0]
        other_end = other.support_start + other.coeffs.shape[0]
        
        sum_start = min(self.support_start, other.support_start)
        sum_end = max(self_end, other_end)

        sum_coeffs = []
        for k in range(sum_start, sum_end):
            res = 0
            
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
        coeffs (list[complex_type]): List of complex coefficients.
        support_start (int): Minimum degree that appears in the polynomial.
    """

    def __init__(self, coeffs: list[complex_type], support_start: int = 0):
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
        return self.coeffs.shape[0] - 1

    def conjugate(self):
        r"""Returns the conjugate polynomial on the unit circle. If :math:`p(z) = \sum_k p_k z^k`, then its conjugate is defined as :math:`p^*(z) = \sum_k p_k^* z^{-k}`

        Returns:
            Polynomial: The conjugate polynomial.
        """
        conj_coeffs = [np.conj(x) for x in reversed(self.coeffs)]
        return Polynomial(conj_coeffs, -(self.support_start + self.coeffs.shape[0] - 1))
    
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

        # Pad so they end up with the same length
        coeffs_a = np.fft.fft(np.pad(self.coeffs, pad_width=(0, other.coeffs.shape[0] - 1)))
        coeffs_b = np.fft.fft(np.pad(other.coeffs, pad_width=(0, self.coeffs.shape[0] - 1)))

        # Multiply in the Fourier domain
        coeffs_c = [a * b for a, b in zip(coeffs_a, coeffs_b)]

        # Inverse FFT to get the result
        new_coeffs = np.fft.ifft(coeffs_c)
        support_start = self.support_start + other.support_start  # Lowest degree of the new poly

        return Polynomial(new_coeffs, support_start)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Number):
            return Polynomial([c / other for c in self.coeffs], self.support_start)
        
        raise TypeError("Polynomial division is only possible with scalars.")

    def __call__(self, z) -> complex_type:
        """Evaluates the polynomial using Horner's method.

        Args:
            z (complex): The point at which to evaluate the polynomial.

        Returns:
            complex: The evaluated result.
        """
        res = self.coeffs[-1]
        for k in reversed(range(self.coeffs.shape[0] - 1)):
            res = res * z + self.coeffs[k]
        return res * (z ** self.support_start)

    def eval_at_roots_of_unity(self, N: int) -> list[complex_type]: # TODO remove power-of-two assumption
        """Evaluates the polynomial at the N-th roots of unity using the inverse FFT.

        Args:
            N (int): A power of two specifying the number of roots. If N is not a power of two, then the next power of two is taken.

        Returns:
            list[complex]: List of evaluations at the N-th roots of unity.
            The k-th element will be `self[w^k]`, where `w` is the main N-th root of unity.
        """
        N = next_power_of_two(N)
        M = next_power_of_two(max(N, self.coeffs.shape[0]))

        coeffs = coeffs_pad(self.coeffs, M)
        coeffs = sequence_shift(coeffs, self.support_start)
        # This has the effect of having everything multiplied by z^s

        evals = np.fft.ifft(coeffs, norm='forward') # M evaluations at the M-th roots of unity
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
        return self.truncate(0, self.support_start + self.coeffs.shape[0] - 1)

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
    def __init__(self, c: list[complex_type] | Polynomial):
        if isinstance(c, list) or isinstance(c, np.ndarray):
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
        return self.coeffs.shape[0] - 1

    def __call__(self, x: float_type) -> complex_type:
        """Evaluates the Chebyshev expansion at the given number.

        Args:
            x (real): The point at which to evaluate the expansion.

        Returns:
            complex: The evaluated result.
        """
        theta = np.arccos(x)

        return sum(self[k] * np.cos(k * theta) for k in self.support())
    
    def to_laurent(self):
        """Returns the Laurent polynomial `P(z) = self((z + z^(-1))/2)`."""
        P = Polynomial(np.concat([np.flipud(self.coeffs), self.coeffs[1:]]), support_start=-self.coeffs.shape[0]+1)
        P[0] *= 2
        return P/2
    
    @classmethod
    def from_polynomial(cls, P: Polynomial):
        """Returns the Chebyshev expansion `T` satisfying `T(x) = P(x)`."""
        return ChebyshevTExpansion(np.polynomial.chebyshev.poly2cheb(P.coeffs))
    
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
        return Polynomial(np.polynomial.chebyshev.cheb2poly(self.coeffs))