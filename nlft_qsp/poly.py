import numpy as np

from numbers import Number

from . import numerics as bd
from .numerics import complex_type, float_type

from .util import coeffs_pad, next_power_of_two, sequence_shift


class ComplexL0Sequence:
    """Represents a sequence of complex numbers or complex matrices index by Z, whose support is finite.
    
    Attributes:
        coeffs (list[complex_type]): List of complex coefficients.
        support_start (int): Index of the first element of the sequence.
        shape (tuple[int]): The shape of the coefficients. This is always () for polynomials, but is used for matrix polynomials.
    """

    def __init__(self, coeffs: list[complex_type] | np.ndarray = None, support_start: int = 0, shape: tuple[int] = None):
        """Initializes a complex sequence.

        Args:
            coeffs: List of complex numbers as coefficients. coeffs[k, :] will be the coefficient of z^(support_start + k).
            shape: The shape of the sequence. If it is none, then the shape of coeffs is taken. This is only used for matrix sequences, and is ignored otherwise.
            support_start (optional): Index of the first element of the sequence. Defaults to 0.

        Note:
            Please provide exactly one of coeffs or shape.
        """
        if coeffs is not None and shape is not None or coeffs is None and shape is None:
            raise ValueError("Please provide exactly one between coeffs or shape.")
        
        self.support_start = support_start
        
        if shape is not None:
            self.coeffs = np.zeros((1, *shape), dtype=complex_type)
            self.shape = shape
            return

        if isinstance(coeffs, list):
            if any(not isinstance(ck, Number) for ck in coeffs):
                raise ValueError(f"The list of coefficients must be a single list or np.ndarray of complex values. Got {coeffs}. If you want to create a matrix sequence, please provide a np.ndarray of shape (num_coeffs, *shape) or specify the shape argument.")
            
            self.coeffs = np.array(coeffs, dtype=complex_type)
            self.shape = ()
            return
        
        if isinstance(coeffs, np.ndarray):
            if coeffs.ndim == 0:
                raise ValueError(f"There must be at least one coefficient. Got {coeffs}.")

            self.coeffs = np.array(coeffs, dtype=complex_type)
            self.shape = coeffs.shape[1:]
            return

        raise ValueError(f"Coefficients must be provided as a list or np.ndarray. Got {coeffs}.")

    def support(self) -> range:
        """Returns the range in Z where the sequence is non-zero.

        Note:
            This simply checks the allocated array of coefficients but does not check leading or trailing zeros.

        Returns:
            range: The support of the sequence.
        """
        return range(self.support_start, self.support_start + self.coeffs.shape[0])
    
    def __getitem__(self, k: int | tuple) -> complex_type | np.ndarray:
        """Returns the k-th element of the sequence, i.e., F_k.

        Args:
            k (int): The index of the sequence.

        Returns:
            complex_type | np.ndarray: The coefficient of F_k, or 0 if k is out of the support.

        Note: in case of matrix sequence, numpy slices can be used on the coefficients.
        """
        if isinstance(k, int):
            k = (k,)

        if k[0] in self.support():
            return self.coeffs[k[0] - self.support_start, *k[1:]]
        return 0

    def __setitem__(self, k: int, c: complex_type | np.ndarray):
        """Sets the coefficient of z^k to be c, allocating space if needed.

        Args:
            k (int): The exponent of z.
            c (complex): The coefficient to set.
        """
        if isinstance(k, int):
            k = (k,)

        if self.support_start + self.coeffs.shape[0] <= k[0]:
            pad_width = [(0, k[0] - self.support_start - self.coeffs.shape[0] + 1)] + [(0, 0)] * (self.coeffs.ndim - 1)
            self.coeffs = np.pad(self.coeffs, pad_width)
        elif self.support_start > k[0]:
            pad_width = [(self.support_start - k[0], 0)] + [(0, 0)] * (self.coeffs.ndim - 1)
            self.coeffs = np.pad(self.coeffs, pad_width)
            self.support_start = k[0]

        self.coeffs[k[0] - self.support_start, *k[1:]] = c

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

        Note:
            For matrix sequences, this is the sum of the squared Frobenius norms of the coefficients.
        """
        if self.shape == ():
            return sum(c * np.conj(c) for c in self.coeffs)
        
        return sum(np.linalg.norm(c)**2 for c in self.coeffs)
    
    def is_real(self) -> bool:
        """Whether the sequence has only real elements."""
        if self.shape == ():
            return all(np.abs(np.imag(F)) <= bd.machine_threshold() for F in self.coeffs)
        
        return all(np.abs(np.imag(F)) <= bd.machine_threshold() for F in self.coeffs.flatten())
    
    def is_imaginary(self) -> bool:
        """Whether the sequence has only imaginary elements."""
        if self.shape == ():
            return all(np.abs(np.real(F)) <= bd.machine_threshold() for F in self.coeffs)
        
        return all(np.abs(np.real(F)) <= bd.machine_threshold() for F in self.coeffs.flatten())
    
    def is_symmetric(self) -> bool:
        """Whether the sequence satisfies F[k] = F[-k]."""
        if self.shape == ():
            return all(np.abs(self[k] - self[-k]) <= bd.machine_threshold() for k in self.support())
        
        return all(np.linalg.norm(self[k] - self[-k]) <= bd.machine_threshold() for k in self.support())
    
    def __add__(self, other):
        if isinstance(other, Number):
            q = self.duplicate()
            q[0] += other

            return q
        elif isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise ValueError(f"The shape of the given matrix {other.shape} does not match the expected shape {self.shape}.")

            q = self.duplicate()
            q[0] += other

            return q
        elif not isinstance(other, ComplexL0Sequence):
            raise TypeError("Sequence addition admits only other sequences or scalars.")
                
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
            
        return Polynomial(np.array(sum_coeffs, dtype=complex_type), sum_start) # TODO This should return the lowest class between self and other.
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return type(self)(-self.coeffs, self.support_start)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)


class Polynomial(ComplexL0Sequence):
    """Represents a general Laurent polynomial of one complex variable.

    Attributes:
        coeffs (list[complex_type]): List of complex coefficients.
        shape (tuple): The shape of the coefficients. This is always () for polynomials, but is used for matrix polynomials.
        support_start (int): Minimum degree that appears in the polynomial.
    """

    def __init__(self, coeffs: list[complex_type] | np.ndarray = None, support_start: int = 0, shape: tuple[int] = None):
        """Initializes a Polynomial instance.

        Args:
            coeffs: List of complex numbers as coefficients.
            shape (optional): Shape of the coefficient array.
            support_start (optional): Minimum degree in the polynomial. Defaults to 0.

        Note:
            Please provide exactly one of coeffs or shape.
        """
        super().__init__(coeffs, support_start, shape)

    def duplicate(self):
        """Creates a duplicate of the current polynomial.

        Returns:
            Polynomial: A new Polynomial instance with the same coefficients and support.
        """
        return type(self)(self.coeffs, self.support_start)
    
    def shift(self, k: int):
        """Creates a new polynomial equal to the current one, multiplied by `z^k`."""
        return type(self)(self.coeffs, self.support_start + k)

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

        Note: for a matrix polynomial, each coefficient is conjugate-transposed.
        """
        if self.shape == ():
            conj_coeffs = np.flip(np.conj(self.coeffs), axis=0)
        else:
            axes = [0] + list(range(1, self.coeffs.ndim))[::-1] # transpose = reverse the order of the axes except the first one, which is the index of the coefficient
            conj_coeffs = np.flip(np.transpose(np.conj(self.coeffs), axes=axes), axis=0)
        
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

        return Polynomial(np.array(schwarz_coeffs, dtype=complex_type), self.support_start)
    
    def hilbert_transform(self):
        r"""Returns the polynomial P such that P + self yields an analytic polynomial.

        Note: This is actually i H[self], i.e., the Hilbert transform as returned is already multiplied by i.
        
        Returns:
            Polynomial: The Hilbert transform of the polynomial.
        """
        hilbert_coeffs = []
        for k in self.support():
            if k < 0:
                hilbert_coeffs.append(-self[k])
            elif k > 0:
                hilbert_coeffs.append(self[k])
            else:
                hilbert_coeffs.append(0)

        return Polynomial(np.array(hilbert_coeffs, dtype=complex_type), self.support_start)

    def __mul__(self, other):
        if isinstance(other, Number):
            return Polynomial(self.coeffs * other, self.support_start)
        elif isinstance(other, np.ndarray):
            if other.shape != self.shape:
                raise ValueError(f"The shape of the given matrix {other.shape} does not match the expected shape {self.shape}.")

            return Polynomial(self.coeffs * other, self.support_start)
        elif not isinstance(other, Polynomial):
            raise TypeError("Polynomial multiplication admits only other polynomials or constants with compatible shape.")

        # Pad so they end up with the same length
        target_len = self.coeffs.shape[0] + other.coeffs.shape[0] - 1
        pad_a = [(0, target_len - self.coeffs.shape[0])] + [(0, 0)] * (self.coeffs.ndim - 1)
        pad_b = [(0, target_len - other.coeffs.shape[0])] + [(0, 0)] * (other.coeffs.ndim - 1)

        coeffs_a = np.fft.fft(np.pad(self.coeffs, pad_width=pad_a), axis=0)
        coeffs_b = np.fft.fft(np.pad(other.coeffs, pad_width=pad_b), axis=0)

        # Multiply in the Fourier domain
        coeffs_c = [a * b for a, b in zip(coeffs_a, coeffs_b)]

        # Inverse FFT to get the result
        new_coeffs = np.fft.ifft(coeffs_c, axis=0)
        support_start = self.support_start + other.support_start  # Lowest degree of the new poly

        return Polynomial(new_coeffs, support_start)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Number):
            return Polynomial(self.coeffs / other, self.support_start)
        
        raise TypeError("Polynomial division is only possible with scalars.")
    
    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            if self.coeffs.shape[-1] != other.shape[-2]:
                raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}.")

            return Polynomial(self.coeffs @ other, self.support_start)
        elif not isinstance(other, Polynomial):
            raise TypeError("Polynomial multiplication admits only other polynomials or constants with compatible shape.")

        if self.coeffs.shape[-1] != other.shape[-2]:
            raise ValueError(f"Incompatible shapes: {self.shape} vs {other.shape}.")

        # Pad so they end up with the same length
        target_len = self.coeffs.shape[0] + other.coeffs.shape[0] - 1
        pad_a = [(0, target_len - self.coeffs.shape[0])] + [(0, 0)] * (self.coeffs.ndim - 1)
        pad_b = [(0, target_len - other.coeffs.shape[0])] + [(0, 0)] * (other.coeffs.ndim - 1)

        coeffs_a = np.fft.fft(np.pad(self.coeffs, pad_width=pad_a), axis=0)
        coeffs_b = np.fft.fft(np.pad(other.coeffs, pad_width=pad_b), axis=0)

        # Multiply in the Fourier domain
        coeffs_c = [a @ b for a, b in zip(coeffs_a, coeffs_b)]

        # Inverse FFT to get the result
        new_coeffs = np.fft.ifft(coeffs_c, axis=0)
        support_start = self.support_start + other.support_start  # Lowest degree of the new poly

        return Polynomial(new_coeffs, support_start)
    
    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            if other.shape[-1] != self.coeffs.shape[-2]:
                raise ValueError(f"The shape of the given matrix {other.shape} does not match the expected shape {self.shape}.")

            return Polynomial(other @ self.coeffs, self.support_start)

        raise TypeError("Polynomial multiplication admits only other polynomials or constants with compatible shape.")

    def __call__(self, z) -> complex_type | np.ndarray:
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

        pad_width = [(0, M - self.coeffs.shape[0])] + [(0, 0)] * (self.coeffs.ndim - 1)
        coeffs = np.pad(self.coeffs, pad_width=pad_width)
        coeffs = np.roll(coeffs, self.support_start, axis=0)
        # This has the effect of having everything multiplied by z^s

        evals = np.fft.ifft(coeffs, norm='forward', axis=0) # M evaluations at the M-th roots of unity
        return evals[::M//N]
    
    def sup_norm(self, N=1024):
        """Estimates the supremum norm of the polynomial over the unit circle
        
        Args:
            N (int, optional): the number of samples to compute the maximum from. If N is not a power of two, then the next power of two is taken.

        Returns:
            float: An estimate for the supremum norm of the polynomial over the unit circle.
        """
        if self.shape == ():
            return np.max([np.abs(sample) for sample in self.eval_at_roots_of_unity(N)])
        
        return np.max([np.linalg.norm(sample, ord=2) for sample in self.eval_at_roots_of_unity(N)])
    
    def truncate(self, m: int, n: int):
        """Keeps only the coefficients in [m, n], discarding the others.

        Returns:
            Polynomial: A new, truncated polynomial.
        """
        if self.shape == ():
            return Polynomial([self[k] for k in range(m, n+1)], m)
        
        return Polynomial(np.array([self[k] for k in range(m, n+1)], dtype=complex_type), m)
    
    def only_positive_degrees(self):
        """Discards all the negative degrees, keeping only the non-negative ones.
        
        Returns:
            Polynomial: A new polynomial containing only the positive-degree coefficients."""
        return self.truncate(0, self.support_start + self.coeffs.shape[0] - 1)
    
    def only_negative_degrees(self):
        """Discards all the positive degrees, keeping only the non-positive ones.
        
        Returns:
            Polynomial: A new polynomial containing only the negative-degree coefficients."""
        return self.truncate(self.support_start, 0)

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