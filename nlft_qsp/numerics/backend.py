
import functools

from typing import SupportsFloat, SupportsComplex, TypeAlias


"""Type alias for generic floating-point real numbers.
"""
generic_real: TypeAlias = SupportsFloat | float

"""Type alias for generic floating-point complex numbers.
"""
generic_complex: TypeAlias = SupportsComplex | complex


class DummyPrecisionManager:
    """This replaces mpmath's precision manager for those backend interfaces
    that do not support variable-precision arithmetic."""

    def __call__(self, f):
        @functools.wraps(f)
        def g(*args, **kwargs):
            return f(*args, **kwargs)

        return g
    
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class NumericBackend:
    """This class provides an interface for various mathematical functions used in the package.
    The can be implemented by different libraries for numerical computations, so that the package
    can benefit from either arbitrary precision arithmetic, or fast, hardware accelerated
    fixed-precision arithmetic."""

    def pi(self):
        return 3.14

    def machine_eps(self):
        """Returns the machine epsilon, as a float object of the backend."""
        raise NotImplementedError()
    
    def machine_threshold(self):
        """Returns the threshold of the backend. Any number under this threshold can be chopped to zero."""
        raise NotImplementedError()
    
    def workdps(self, x: int):
        """Temporarily sets the working precision to the given value (in dps).
        This method does not do anything if the backend has fixed precision.

        Note:
            To be used in a `with` statement or as a function decorator.
            This method does not do anything if the backend has fixed precision."""
        return DummyPrecisionManager()
    
    def workprec(self, x: int):
        """Temporarily sets the working precision to the given value (in bits).
        This method does not do anything if the backend has fixed precision.

        Note:
            To be used in a `with` statement or as a function decorator.
            This method does not do anything if the backend has fixed precision."""
        return DummyPrecisionManager()
    
    def extradps(self, x: int):
        """Temporarily increases the working precision by the given amount (in dps).

        Note:
            To be used in a `with` statement or as a function decorator.
            This method does not do anything if the backend has fixed precision."""
        return DummyPrecisionManager()
    
    def extraprec(self, x: int):
        """Temporarily increases the working precision by the given amount (in bits).
        This method does not do anything if the backend has fixed precision.
        
        Note:
            To be used in a `with` statement or as a function decorator.
            This method does not do anything if the backend has fixed precision."""
        return DummyPrecisionManager()
    