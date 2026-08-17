"""Module dealing with floating point types and error tolerance."""

from typing import TypeAlias

import numpy as np

"""Type used throughout the package for complex numbers."""
complex_type: TypeAlias = np.complex128

"""Type used throughout the package for real numbers."""
float_type: TypeAlias = np.float64


"""Smallest possible error possible with the working floating-point type."""
def machine_eps():
    return np.finfo(complex_type).eps

"""Threshold under which any number should be treated as zero"""
def machine_threshold():
    return 1e-8 # TODO make tunable