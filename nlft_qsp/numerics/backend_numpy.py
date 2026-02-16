
import numpy as np
import scipy.linalg


from .backend import NumericBackend
from .backend import generic_complex, generic_real

from ..util import next_power_of_two

def select_largest_dtypes():
    """Select the largest dtype available in the platform."""
    ctypes = ['complex512', 'complex256', 'complex192', 'complex160', 'complex128']
    ftypes = ['float256', 'float128', 'float96', 'float80', 'float64']

    for ct, ft in zip(ctypes, ftypes):
        if hasattr(np, ct):
            return getattr(np, ct), getattr(np, ft)
    
    return complex, float


class NumpyBackend(NumericBackend):
    """Numeric backend for numpy.
    
    Note:
        Using numpy allows to take advantage to optimized arithmetic computation,
        but it gives poor precision.
    """

    def __init__(self, verbose=False):
        """Initializes a numpy backend.
        """
        dtype, ftype = select_largest_dtypes()
        
        if verbose:
            print('NumpyBackend -- chosen dtypes: %s, %s' % (dtype.__name__, ftype.__name__))

        self.dtype = dtype
        self.ftype = ftype

    def __getattr__(self, item): # redirect any other function to numpy
        return getattr(np, item)

    def pi(self):
        return np.pi

    def machine_eps(self):
        return np.finfo(self.dtype).eps
    
    def machine_threshold(self):
        return 1e-8
    
    def chop(self, x: generic_complex):
        thr = self.machine_threshold()

        if np.abs(x.real) < thr:
            x.real = 0

        if np.abs(x.imag) < thr:
            x.imag = 0

        return x

    def make_complex(self, x: generic_complex):
        return self.dtype(x)

    def make_float(self, x: generic_real):
        return self.ftype(x)

    def abs2(self, x: generic_complex):
        return np.real(x) ** 2 + np.imag(x) ** 2
    
    def re(self, x: generic_complex):
        return np.real(x)
    
    def im(self, x: generic_complex):
        return np.imag(x)
    
    def arg(self, x: generic_complex):
        return np.angle(x) # TODO this only works with float64
    
    def unitroots(self, N: int):
        return [np.exp(2j*np.pi*k/N) for k in range(N)]

    def fft(self, x: list[generic_complex], normalize=False):
        N = len(x)
        M = next_power_of_two(N)

        if M > N:
            x = x + [self.make_complex(0)] * (M - N)

        if normalize:
            norm = 'forward'
        else:
            norm = 'backward'

        return np.fft.fft(np.array(x), norm=norm).tolist()
    
    def ifft(self, x: list[generic_complex], normalize=True):
        N = len(x)
        M = next_power_of_two(N)

        if M > N:
            x = x + [self.make_complex(0)] * (M - N)

        if normalize:
            norm = 'backward'
        else:
            norm = 'forward'

        return np.fft.ifft(np.array(x), norm=norm).tolist()
    
    def fft_md(self, x: list, shape: tuple=None, normalize=False):
        if normalize:
            norm = 'forward'
        else:
            norm = 'backward'

        return np.fft.fftn(np.array(x), shape, norm=norm).tolist()
    
    def ifft_md(self, x: list, shape: tuple=None, normalize=True):
        if normalize:
            norm = 'backward'
        else:
            norm = 'forward'

        return np.fft.ifftn(np.array(x), shape, norm=norm).tolist()
    
    def matrix(self, x: list):
        return np.array(x, dtype=self.dtype)
    
    def to_list(self, x):
        return x.tolist()
    
    def transpose(self, x):
        return np.transpose(x)
    
    def conj_transpose(self, x):
        return np.transpose(np.conjugate(x))
    
    def zeros(self, m: int, n: int):
        return np.zeros(shape=(m, n), dtype=self.dtype)
    
    def eye(self, n: int):
        return np.eye(n, dtype=self.dtype)
    
    def solve_system(self, A, b):
        return np.linalg.solve(A, b)
    
    def qr_decomp(self, A):
        return scipy.linalg.qr(A)
    
    def poly2cheb(self, p):
        return np.polynomial.chebyshev.poly2cheb(p).tolist()
    
    def cheb2poly(self, p):
        return np.polynomial.chebyshev.cheb2poly(p).tolist()