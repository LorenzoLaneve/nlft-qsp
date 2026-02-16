import mpmath as mp

from .backend_mpmath import MPMathBackend
from .backend_numpy import NumpyBackend

__all__ = [
    "MPMathBackend",
    "NumpyBackend",
    "set_backend"
]


"""The currently active backend. This exposes all the mathematical functions needed by the package
and offered by the underlying library."""
class __bd_wrapper:
    # In order to test with a particular backend, the backend has to be temporarily set from here.
    bd = NumpyBackend()
    #bd = MPMathBackend(mp.mp)


def set_backend(new_bd):
    """Sets the current working backend to the given `NumericBackend` object."""
    __bd_wrapper.bd = new_bd

def coeffs_pad(c: list, N: int):
    """Pads the list c with zeros so that it results of length N. If len(c) >= N, then the list will be left unchanged.

    Args:
        c (list[complex]): the list to be padded
        N (int): The length of the padded list.

    Returns:
        list[complex]: The original list padded with zeros, such that the total length will be N.
    """
    if len(c) < N:
        return c + [make_complex(0)] * (N - len(c))

    return c


def pi():
    return __bd_wrapper.bd.pi()

def machine_eps():
    """Returns the machine epsilon, as a float object of the backend."""
    return __bd_wrapper.bd.machine_eps()
    
def machine_threshold():
    """Returns the threshold of the backend. Any number under this threshold can be chopped to zero."""
    return __bd_wrapper.bd.machine_threshold()

def workdps(x: int):
    """Temporarily sets the working precision to the given value (in dps).
    This method does not do anything if the backend has fixed precision.

     Note:
        To be used in a `with` statement or as a function decorator.
        This method does not do anything if the backend has fixed precision."""
    return __bd_wrapper.bd.workdps(x)
    
def workprec(x: int):
    """Temporarily sets the working precision to the given value (in bits).
    This method does not do anything if the backend has fixed precision.
    
    Note:
        To be used in a `with` statement or as a function decorator.
        This method does not do anything if the backend has fixed precision."""
    return __bd_wrapper.bd.workprec(x)
    
def extradps(x: int):
    """Temporarily increases the working precision by the given amount (in dps).

    Note:
        To be used in a `with` statement or as a function decorator.
        This method does not do anything if the backend has fixed precision."""
    return __bd_wrapper.bd.extradps(x)
    
def extraprec(x: int):
    """Temporarily increases the working precision by the given amount (in bits).
    This method does not do anything if the backend has fixed precision.
        
    Note:
        To be used in a `with` statement or as a function decorator.
        This method does not do anything if the backend has fixed precision."""
    return __bd_wrapper.bd.extraprec(x)
    
def chop(x):
    """Returns the same number, or 0 if `abs(x)` goes below the machine threshold."""
    return __bd_wrapper.bd.chop(x)

def make_complex(x):
    """Construct the given complex number as an object of the backend."""
    return __bd_wrapper.bd.make_complex(x)
    
def make_float(x):
    """Construct the given real number as an object of the backend."""
    return __bd_wrapper.bd.make_float(x)

def abs(x):
    """Returns the absolute value of the given complex number."""
    return __bd_wrapper.bd.abs(x)
    
def abs2(x):
    """Returns the absolute value squared of the given complex number."""
    return __bd_wrapper.bd.abs2(x)
    
def sqrt(x):
    """Returns the square root of the given complex number.
    Note:
        The chosen root depends on the implementation of the backend.
    """
    return __bd_wrapper.bd.sqrt(x)
    
def log(x):
    """Returns the natural logarithm of the given complex number."""
    return __bd_wrapper.bd.log(x)
    
def exp(x):
    """Returns the exponential of the given complex number."""
    return __bd_wrapper.bd.exp(x)

def sin(x):
    """Returns the sine of the given complex number."""
    return __bd_wrapper.bd.sin(x)

def cos(x):
    """Returns the cosine of the given complex number."""
    return __bd_wrapper.bd.cos(x)

def tan(x):
    """Returns the tangent of the given complex number."""
    return __bd_wrapper.bd.tan(x)

def arcsin(x):
    """Returns the arcsine of the given complex number."""
    return __bd_wrapper.bd.arcsin(x)

def arccos(x):
    """Returns the arccosine of the given complex number."""
    return __bd_wrapper.bd.arccos(x)

def arctan(x):
    """Returns the arctangent of the given complex number."""
    return __bd_wrapper.bd.arctan(x)

def re(x):
    """Returns the real part of the given complex number."""
    return __bd_wrapper.bd.re(x)

def im(x):
    """Returns the imaginary part of the given complex number."""
    return __bd_wrapper.bd.im(x)
    
def conj(x):
    """Returns the conjugate of the given complex number."""
    return __bd_wrapper.bd.conj(x)

def arg(x):
    """Returns the argument of the given complex number, between -pi and pi."""
    return __bd_wrapper.bd.arg(x)
    
def unitroots(N: int):
    """Returns a list containing the N-th roots of unity, as objects of the backend."""
    return __bd_wrapper.bd.unitroots(N)
    
def fft(x: list, normalize=False):
    """Computes the Fast Fourier Transform of the given list of complex numbers.
    The list is padded to the next power of two.
        
    Args:
        normalize (bool): whether the result should be divided by the length of the vector."""
    return __bd_wrapper.bd.fft(x, normalize)
    
def ifft(x: list, normalize=True):
    """Computes the Inverse Fast Fourier Transform of the given list of complex numbers.
    The list is padded to the next power of two.
        
    Args:
        normalize (bool): whether the result should be divided by the length of the vector."""
    return __bd_wrapper.bd.ifft(x, normalize)

def fft_md(x: list, shape: tuple=None, normalize=False):
    """Computes the Fast Fourier Transform in multiple dimensions of the given list of complex numbers.
    The list is padded to the next power of two in each dimension.
        
    Args:
        normalize (bool): whether the result should be divided by the size of the list."""
    return __bd_wrapper.bd.fft_md(x, shape, normalize)
    
def ifft_md(x: list, shape: tuple=None, normalize=True):
    """Computes the Inverse Fast Fourier Transform in multiple dimensions of the given list of complex numbers.
    The list is padded to the next power of two in each dimension.
        
    Args:
        normalize (bool): whether the result should be divided by the size of the list."""
    return __bd_wrapper.bd.ifft_md(x, shape, normalize)
    
def matrix(x: list):
    """Constructs an object of the backend representing a matrix with the given list (of lists) of coefficients."""
    return __bd_wrapper.bd.matrix(x)

def to_list(x):
    """Returns the matrix given as an object of the backend as a Python list, containing objects of the backend."""
    return __bd_wrapper.bd.to_list(x)
    
def transpose(x):
    """Returns the transpose of the given matrix. Both input and output are given as objects of the backend."""
    return __bd_wrapper.bd.transpose(x)
    
def conj_transpose(x):
    """Returns the conjugate transpose of the given matrix. Both input and output are given as objects of the backend."""
    return __bd_wrapper.bd.conj_transpose(x)
    
def zeros(m: int, n: int):
    """Constructs a `m x n` zero matrix, as an object of the backend."""
    return __bd_wrapper.bd.zeros(m, n)

def eye(n: int):
    """Constructs the `n x n` identity matrix as an object of the backend."""
    return __bd_wrapper.bd.eye(n)
    
def solve_system(A, b):
    """Solves the linear system `Ax = b` and returns x as a list.
    The list may be returned as an object of the backend."""
    return __bd_wrapper.bd.solve_system(A, b)
    
def qr_decomp(A):
    """Decomposes A = QR, where Q is unitary and R is upper triangular.
    The matrix in input, as well as the two outputs, are given as objects of the backend."""
    return __bd_wrapper.bd.qr_decomp(A)

def poly2cheb(p):
    """Returns a list of coefficients (as objects of the backend) corresponding to the
    Chebyshev expansion of the polynomial `p`, given as list of coefficients."""
    return __bd_wrapper.bd.poly2cheb(p)

def cheb2poly(c):
    """Returns a list of coefficients (as objects of the backend) corresponding to the
    polynomial of the Chebyshev expansion `c`, given as list of coefficients."""
    return __bd_wrapper.bd.cheb2poly(c)