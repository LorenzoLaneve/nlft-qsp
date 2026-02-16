
import itertools
from math import prod
from numbers import Number
from typing import Iterable

from . import numerics as bd
from .numerics.backend import generic_complex, generic_real

from .poly import Polynomial
from .util import flatten, next_power_of_two, sequence_shift

def minimal_covering_range(l):
    """Given a list of tuples `l` containing `range` objects, returns a tuple `t` containing `range`
    objects such that `t[k] is the minimal range containing `l[j][k]` for every `j`."""
    if len(l) == 0:
        return ()
    
    N = max(len(t) for t in l)

    return tuple(
        range(
            min((rng.start for rng in col if rng is not None), default=0),  
            max((rng.stop for rng in col if rng is not None), default=0)
        )
        for col in zip(*[t + (None,) * (N - len(t)) for t in l])
    )

def deep_truncate(l, lens):
    """Returns the same multidimensional list truncates to the given lengths in each axis.

    Note:
        It is assumed that l has dimensions >= lens.
    
    Args:
        lens (tuple[int])"""
    if len(lens) > 1:
        ilens = lens[1:]
        return [deep_truncate(l[k], ilens) for k in range(lens[0])]
    
    return l[:lens[0]]

def deep_horner_eval(l, z: tuple[generic_complex]):
    """Evaluates the multivariate polynomial with coefficients l using Horner's method."""
    if len(z) == 0:
        return l
    
    evals = [deep_horner_eval(il, z[1:]) for il in l]
    
    res = evals[-1]
    for k in reversed(range(len(evals) - 1)):
        res = res * z[0] + evals[k]

    return res

def deep_sequence_shift(l, s: tuple[int]):
    """Cyclically shifts the elements along each axis, where k-th axis is shifted by k."""
    if isinstance(l, list) and len(l) != 0:
        if isinstance(l[0], Number):
            return sequence_shift(l, s[0])
    
    s1 = s[1:]
    return sequence_shift([deep_sequence_shift(il, s1) for il in l], s[0])

def deep_sequence_skip(l, s: tuple[int]):
    """Takes only every `s[k]` element of the list in each axis, the equivalent of l[::s] in multiple dimensions."""
    if isinstance(l, list) and len(l) != 0:
        if isinstance(l[0], Number):
            return l[::s[0]]
    
    s1 = s[1:]
    l = l[::s[0]]
    return [deep_sequence_skip(il, s1) for il in l]

def deep_inplace(l, func, reverse=False):
    """Applies the function to each element of the given nested list, in place."""
    if isinstance(l, list) and len(l) != 0:
        if reverse:
            l.reverse()

        if isinstance(l[0], Number):
            for k in range(len(l)):
                l[k] = func(l[k])
        else:
            for il in l:
                deep_inplace(il, func, reverse)

def deep_inplace_binary(l1, l2, func):
    """Applies the binary function to each element of the given nested lists, in place (they are assumed to be of the same dimension)."""
    if isinstance(l1, list) and len(l1) != 0:

        if isinstance(l1[0], Number):
            for k in range(len(l1)):
                l1[k] = func(l1[k], l2[k])
        else:
            for il1, il2 in zip(l1, l2):
                deep_inplace_binary(il1, il2, func)

def zeros(lens):
    if len(lens) == 1:
        return [bd.make_complex(0)] * (lens[0])
    
    zr = []
    for k in range(lens[0]):
        zr.append(zeros(lens[1:]))

    return zr

def sgn(k: int):
    if k > 0:
        return 1
    if k < 0:
        return -1
    return 0

def schwarz_multiplier(k: tuple):
    """Fourier multiplier for the Schwarz transform over several variables."""
    if all(c == 0 for c in k):
        return 1
    
    if all(c <= 0 for c in k):
        return 2
    
    return 0


class ComplexL0SequenceMD:

    def __init__(self, coeffs: list, support_start: tuple[int] | int):
        if isinstance(support_start, int):
            support_start = (support_start,)

        self.dim = len(support_start)
        self._xsupport_start = support_start[0]

        if not isinstance(coeffs, list):
            raise ValueError("Coefficient list must be of type list.")

        if self.dim == 1:
            if not all(isinstance(c, Number) for c in coeffs):
                raise ValueError("Coefficient list must be of the corresponding dimension.")
            
            self.coeffs = [bd.make_complex(c) for c in coeffs]
        else:
            self.coeffs = []

            for row in coeffs:
                if isinstance(row, list):
                    self.coeffs.append(self.__class__(row, support_start[1:]))
    
    def __xsupport(self):
        return range(self._xsupport_start, self._xsupport_start + len(self.coeffs))

    def support(self) -> tuple[range]:
        """Returns a tuple containing support ranges for all dimensions, i.e.,
        the hyper-parallelepiped in the grid containing all the coefficients of the polynomial."""
        xsupp = self.__xsupport()

        if self.dim == 1:
            return (xsupp,)
        
        return (xsupp, *minimal_covering_range([c.support() for c in self.coeffs]))
    
    @property
    def support_start(self):
        return tuple([t.start for t in self.support()])
    
    def coeff_list(self, rng=None):
        r"""Returns a `dim`-dimensional list containing the coefficients.

        Note:
            The stop element of the ranges are not included in the list.
        
        Args:
            rng: `dim`-dimensional tuple of range objects, giving the range for the hyper-parallelepiped along each axis. Defaults to the support of the sequence, as returned by `support()`.
        """
        if rng is None:
            rng = self.support()

        if self.dim == 1:
            return [self[k] for k in rng[0]]
        
        hcube = []
        lens = tuple(r.stop - r.start for r in rng[1:])
        for k in range(rng[0].start, rng[0].stop):

            if k - self._xsupport_start in range(len(self.coeffs)):
                hcube.append(self.coeffs[k - self._xsupport_start].coeff_list(rng[1:]))
            else:
                hcube.append(zeros(lens))
        
        return hcube
    
    def coeff_dict(self) -> dict:
        """Returns a dictionary of values representing the flattened version of `coeff_list()`.
        The keys are m-dimensional tuples, and the values are the corresponding coefficients."""
        L = self.coeff_list()
        result = {}

        def recurse(sublist, index_prefix=()):
            if not isinstance(sublist, list):
                # reached a coefficient value
                result[index_prefix] = sublist
                return
            for i, elem in enumerate(sublist):
                recurse(elem, index_prefix + (i,))

        recurse(L)
        return result
    
    def duplicate(self):
        return self.__class__(self.coeff_list(), self.support_start)
    
    def __getitem__(self, k: int) -> generic_complex:
        """Returns the coefficient in position (k1, ..., kd), or zero if the element is outside the support.
        """
        if not isinstance(k, tuple):
            k = (k,)

        if len(k) != self.dim:
            raise ValueError("Number of indices must coincide with the dimension of the sequence.")
        
        x = k[0]
        if x in self.__xsupport():
            if self.dim == 1:
                return self.coeffs[x - self._xsupport_start]
            
            return self.coeffs[x - self._xsupport_start][k[1:]]
        
        return bd.make_complex(0)

    def __setitem__(self, k: int, c: generic_complex):
        """Sets the coefficient of (k1, ..., kd) to be c, allocating space if needed.
        """
        x = k[0]
        if self.dim == 1:
            if self._xsupport_start + len(self.coeffs) <= x:
                self.coeffs.extend([bd.make_complex(0)] * (x - self._xsupport_start - len(self.coeffs) + 1))
            elif self._xsupport_start > x:
                self.coeffs = [bd.make_complex(0)] * (self._xsupport_start - x) + self.coeffs
                self._xsupport_start = x
            self.coeffs[x - self._xsupport_start] = bd.make_complex(c)
        else:
            if self._xsupport_start + len(self.coeffs) <= x:
                for _ in range(x - self._xsupport_start - len(self.coeffs) + 1):
                    self.coeffs.append(self.__class__([], support_start=(0,) * (self.dim-1)))
            elif self._xsupport_start > x:
                for _ in range(self._xsupport_start - x):
                    self.coeffs = [self.__class__([], support_start=(0,) * (self.dim-1))] + self.coeffs
                self._xsupport_start = x

            self.coeffs[x - self._xsupport_start][k[1:]] = bd.make_complex(c)

    def l1_norm(self) -> generic_real:
        """Computes the l1 norm of the sequence.

        Returns:
            float: The sum of absolute values of coefficients.
        """
        if self.dim == 1:
            return sum(bd.abs(c) for c in self.coeffs)
        return sum(c.l1_norm() for c in self.coeffs)

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
        if self.dim == 1:
            return sum(bd.abs2(c) for c in self.coeffs)
        return sum(c.l2_squared_norm() for c in self.coeffs)
    
    def is_real(self) -> bool:
        """Whether the sequence has only real elements."""
        if self.dim == 1:
            return all(bd.im(F) <= bd.machine_threshold() for F in self.coeffs)
        return all(c.is_real() for c in self.coeffs)
    
    def is_imaginary(self) -> bool:
        """Whether the sequence has only imaginary elements."""
        if self.dim == 1:
            return all(bd.re(F) <= bd.machine_threshold() for F in self.coeffs)
        return all(c.is_imaginary() for c in self.coeffs)
    
    def _coeffwise_unary(self, func):
        """Returns a new sequence object `r` (as an object of the same class as self) whose coefficients are `r[k] = func(self[k])`.
        
        Note:
            It is implicitly assumed that func(0) == 0, so that compactness of the support is preserved."""
        cf = self.coeff_list()
        deep_inplace(cf, func)
        return self.__class__(cf, self.support_start)

    def _coeffwise_binary(self, other, func):
        """Returns a new sequence object `r` (as an object of the same class as self) whose coefficients are the pairwise `r[k] = func(self[k], other[k])`.
        
        Note:
            It is implicitly assumed that func(0, 0) == 0, i.e., the support of `r` will be the union of the supports of `p`, `q`."""
        union_support = tuple(range(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(self.support(), other.support()))
        union_start = tuple(min(x, y) for x, y in zip(self.support_start, other.support_start))

        cf1 = self.coeff_list(union_support)
        cf2 = other.coeff_list(union_support)
        deep_inplace_binary(cf1, cf2, func)

        return self.__class__(cf1, union_start)
    
    def __add__(self, other):
        if isinstance(other, Number):
            q = self.duplicate()
            q[(0,) * self.dim] += other

            return q
        elif not isinstance(other, ComplexL0SequenceMD):
            raise TypeError("Sequence addition admits only other sequences or scalars.")
        
        return self._coeffwise_binary(other, lambda x, y: x + y)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self._coeffwise_unary(lambda x: -x)
    
    def __sub__(self, other):
        if isinstance(other, Number):
            q = self.duplicate()
            q[(0,) * self.dim] -= other

            return q
        elif not isinstance(other, ComplexL0SequenceMD):
            raise TypeError("Sequence addition admits only other sequences or scalars.")
        
        return self._coeffwise_binary(other, lambda x, y: x - y)
    
    def __rsub__(self, other):
        if isinstance(other, Number):
            return other + (-self)
        elif not isinstance(other, ComplexL0SequenceMD):
            raise TypeError("Sequence addition admits only other sequences or scalars.")
        
        return other - self
    


class PolynomialMD(ComplexL0SequenceMD):
    """Represents a general multivariate Laurent polynomial."""

    def __init__(self, coeffs: list, support_start: tuple[int] | int):
        """Initializes a PolynomialMD instance.

        Args:
            coeffs: List of complex numbers as coefficients.
            support_start: tuple of minimum degrees in the polynomial.
        """
        super().__init__(coeffs, support_start)
    
    def shift(self, k: int, a: int=0):
        """Creates a new polynomial equal to the current one, multiplied by `z_a^k`."""
        t = self.support_start
        t = tuple(t[j] + (j == a) * k for j in range(len(t)))

        return PolynomialMD(self.coeff_list(), t)

    def effective_degree(self) -> int:
        """Returns a tuple containing the effective degrees with respect to each variable (max degree - min degree).

        Note:
            This does not check for leading or trailing zeros in the coefficient array.
        """
        t = self.support()
        return tuple(rng.stop - rng.start - 1 for rng in t)

    def conjugate(self):
        r"""Returns the conjugate polynomial on the unit circle. If :math:`p(z) = \sum_k p_k z^k`, then its conjugate is defined as :math:`p^*(z) = \sum_k p_k^* z^{-k}`

        Returns:
            PolynomialMD: The conjugate polynomial.
        """
        cf = self.coeff_list()
        deep_inplace(cf, lambda x: bd.conj(x), reverse=True)

        return PolynomialMD(cf, tuple(-rng.stop + 1 for rng in self.support()))
    
    def schwarz_transform(self):
        r"""Returns the anti-analytic polynomial whose real part gives the current polynomial.
        
        In other words, this is equivalent to adding :math:`iH[p]`, where :math:`H[p]` is the Hilbert transform of p.

        Returns:
            Polynomial: The Schwarz transform of the polynomial.
        """
        st = self.duplicate() # TODO This method can be more efficient

        for k in itertools.product(*self.support()):
            st[*k] = schwarz_multiplier(k) * st[*k]

        return st
    
    def __mul__(self, other):
        if isinstance(other, Number):
            return self._coeffwise_unary(lambda x: x * other)
        elif not isinstance(other, PolynomialMD):
            raise TypeError("Polynomial addition admits only other polynomials or scalars.")
        
        sup_a = self.support()
        sup_b = other.support()

        len_a = tuple(x.stop - x.start for x in sup_a)
        len_b = tuple(x.stop - x.start for x in sup_b)
        len_c = tuple(la + lb - 1 for la, lb in zip(len_a, len_b))

        rng_a = tuple(range(xa.start, xa.start + next_power_of_two(xc)) for xa, xc in zip(sup_a, len_c))
        rng_b = tuple(range(xb.start, xb.start + next_power_of_two(xc)) for xb, xc in zip(sup_b, len_c))
        # augmented support for a and b so that we can carry out FFT on their coeff_list()

        # TODO use extra precision here
        cf1 = bd.fft_md(self.coeff_list(rng_a))
        cf2 = bd.fft_md(other.coeff_list(rng_b))

        # Multiply in the Fourier domain
        deep_inplace_binary(cf1, cf2, lambda x, y: x * y) # cf1 *= cf2

        # Inverse FFT to get the result, support starts are the sum in each individual variable
        return PolynomialMD(deep_truncate(bd.ifft_md(cf1), len_c), tuple(x.start + y.start for x, y in zip(sup_a, sup_b)))
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Number):
            return self._coeffwise_unary(lambda x: x / other)
        
        raise TypeError("Polynomial division is only possible with scalars.")
    
    def __call__(self, *z) -> generic_complex:
        """Evaluates the polynomial using Horner's method.

        Args:
            z (complex): The point at which to evaluate the polynomial.

        Returns:
            complex: The evaluated result.
        """
        if len(z) == 1 and isinstance(z[0], tuple):
            z = z[0]

        if self.dim != len(z):
            raise ValueError("Incorrect number of variables to evaluate the multivariate polynomial.")

        return deep_horner_eval(self.coeff_list(), z) * prod(zk ** nk for zk, nk in zip(z, self.support_start))

    def eval_at_roots_of_unity(self, N: int | tuple) -> list[generic_complex]:
        """Evaluates the polynomial at the N-th roots of unity using the inverse FFT.

        Args:
            N (int | tuple): A power of two specifying the number of roots. If N is not a power of two, then the next power of two is taken.
                             A tuple of appropriate dimension can be given, where the k-th variable will be computed in the `N[k]`-th roots of unity.

        Returns:
            list[complex]: List of evaluations at the N-th roots of unity.
            The k-th element will be `self[w^k]`, where `w` is the main N-th root of unity.
        """
        if isinstance(N, Number):
            N = (N,) * self.dim

        if self.dim != len(N):
            raise ValueError("The dimension of the grid must match the dimension of the polynomial.")

        sup = self.support()

        N = tuple(next_power_of_two(k) for k in N)
        M = tuple(next_power_of_two(max(k, r.stop - r.start)) for k, r in zip(N, sup))

        coeffs = self.coeff_list(tuple(range(r.start, r.start + k) for k, r in zip(M, sup)))
        coeffs = deep_sequence_shift(coeffs, self.support_start)
        # This has the effect of having everything multiplied by z[k]^s[k] for each k

        evals = bd.ifft_md(coeffs, normalize=False) # M evaluations at the M-th roots of unity
        return deep_sequence_skip(evals, tuple(m//n for m, n in zip(M, N)))
    
    def sup_norm(self, N=1024):
        """Estimates the supremum norm of the polynomial over the unit multitorus.
        
        Args:
            N (int, optional): the number of samples to compute the maximum from. If N is not a power of two, then the next power of two is taken.

        Returns:
            float: An estimate for the supremum norm of the polynomial over the unit circle.
        """
        return max([abs(x) for x in flatten(self.eval_at_roots_of_unity(N))])
    
    def truncate(self, *rng: tuple[range] | tuple[tuple[range]]):
        """Keeps only the coefficients in the hyperrectangle defined by rng, discarding the other.
        
        Note:
            Both ends of the ranges are included.

        Args:
            rng (tuple[range]): Lower bound of degree.

        Returns:
            PolynomialMD: A new, truncated polynomial.
        """
        if len(rng) == 1 and isinstance(rng, tuple):
            rng = rng[0]
        
        rng = tuple(range(r.start, r.stop+1) for r in rng)

        return PolynomialMD(self.coeff_list(rng), tuple(r.start for r in rng))
    
    def __str__(self, tol=bd.machine_threshold()):
        """Converts the polynomial to a human-readable string representation.

        Returns:
            str: The string representation of the polynomial.
        """
        sup_start = self.support_start
        return ' + '.join(f"{c} z^{tuple(x + y for x, y in zip(k, sup_start))}" for k, c in self.coeff_dict().items() if bd.abs(c) > tol)
    

def to_poly_md(p: Polynomial, m: int) -> PolynomialMD:
    """Converts the given univariate polynomial P(z_m) as a m-variate polynomial object representing `P'(z_1, ..., z_m) = P(z_m)`."""

    cf = p.coeffs
    for _ in range(m-1):
        cf = [cf]

    return PolynomialMD(cf, support_start=(0,)*(m-1) + (p.support_start,))