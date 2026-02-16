from numbers import Number

from . import numerics as bd
from .numerics.backend import generic_complex

from .util import next_power_of_two
from .poly import ComplexL0Sequence, Polynomial


def _infer_shape(m):
    """Infer (rows, cols) from a matrix-like object.
    
    Handles:
    - Objects with a .shape attribute (numpy arrays, etc.)
    - Nested lists/sequences
    
    Args:
        m: A matrix-like object.
    
    Returns:
        tuple: (rows, cols) dimensions.
    
    Raises:
        ValueError: If shape cannot be inferred or is not 2D.
    """
    # Try .shape attribute first (numpy arrays, etc.)
    shape = getattr(m, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"matrix must be 2-dimensional, got shape {shape}")
        return int(shape[0]), int(shape[1])
    
    # Fall back to nested list/sequence
    try:
        rows = len(m)
        if rows == 0:
            raise ValueError("matrix cannot be empty")
        cols = len(m[0])
        if cols == 0:
            raise ValueError("matrix rows cannot be empty")
        return rows, cols
    except TypeError as e:
        raise ValueError(f"cannot infer matrix shape: object is not subscriptable") from e
    except (IndexError, AttributeError) as e:
        raise ValueError(f"cannot infer matrix shape: irregular or malformed matrix") from e

class ComplexL0MatrixSequence:
    """Represents a matrix of ComplexL0Sequence objects.
    
    Each entry (i, j) is a ComplexL0Sequence, allowing each matrix element to have 
    its own independent support.
    
    Attributes:
        shape (tuple): (rows, cols) dimensions of the matrix.
        _sequences (dict): Dictionary mapping (i, j) -> ComplexL0Sequence.
    """

    def __init__(self, shape: tuple[int, int]):
        """Initializes a matrix of complex sequences.

        Args:
            shape: Tuple (rows, cols) specifying the matrix dimensions.
        
        Raises:
            ValueError: If shape is not a 2-tuple of positive integers.
        """
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("shape must be a tuple of length 2")
        rows, cols = shape
        if rows <= 0 or cols <= 0:
            raise ValueError("matrix dimensions must be positive")
        
        self.shape = (rows, cols)
        self._sequences = {}  # (i, j) -> ComplexL0Sequence
    
    def _ensure_sequence(self, i: int, j: int) -> ComplexL0Sequence:
        """Get or create the sequence at position (i, j)."""
        if (i, j) not in self._sequences:
            self._sequences[(i, j)] = ComplexL0Sequence([])
        return self._sequences[(i, j)]
    
    def __getitem__(self, key):
        """Multi-level indexing for ComplexL0MatrixSequence.
        
        Supports three modes:
        - P[i, j] returns the ComplexL0Sequence at position (i, j)
        - P[i, j, k] returns the coefficient of z^k in sequence (i, j)
        - P[:, :, k] returns a 2D array of all k-th coefficients (matrix form)
        
        Args:
            key: Either (i, j), (i, j, k), or (:, :, k)
        
        Returns:
            ComplexL0Sequence, generic_complex, or 2D list depending on indexing mode.
        """
        if not isinstance(key, tuple):
            raise TypeError("indexing requires a tuple")
        
        if len(key) == 2:
            # P[i, j] -> ComplexL0Sequence
            i, j = key
            if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
                raise IndexError(f"index ({i}, {j}) out of bounds for shape {self.shape}")
            return self._ensure_sequence(i, j)
        
        elif len(key) == 3:
            i, j, k = key
            
            # P[:, :, k] -> matrix of k-th coefficients
            if i is Ellipsis or (isinstance(i, slice) and i == slice(None)):
                if j is Ellipsis or (isinstance(j, slice) and j == slice(None)):
                    matrix = []
                    for row_idx in range(self.shape[0]):
                        row = []
                        for col_idx in range(self.shape[1]):
                            seq = self._sequences.get((row_idx, col_idx))
                            if seq is None:
                                row.append(bd.make_complex(0))
                            else:
                                row.append(seq[k])
                        matrix.append(row)
                    return matrix
                else:
                    raise IndexError("invalid indexing: use P[:, :, k] for full matrix or P[i, j, k] for single element")
            
            # P[i, j, k] -> single coefficient
            else:
                if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
                    raise IndexError(f"index ({i}, {j}) out of bounds for shape {self.shape}")
                seq = self._sequences.get((i, j))
                if seq is None:
                    return bd.make_complex(0)
                return seq[k]
        
        else:
            raise TypeError("indexing requires 2 or 3 indices")

    def __setitem__(self, key, value):
        """Multi-level assignment for ComplexL0MatrixSequence.
        
        Supports two modes:
        - P[i, j] = seq assigns a ComplexL0Sequence to position (i, j)
        - P[i, j, k] = c assigns the coefficient of z^k in sequence (i, j)
        
        Args:
            key: Either (i, j) or (i, j, k)
            value: A ComplexL0Sequence or generic_complex
        """
        if not isinstance(key, tuple):
            raise TypeError("indexing requires a tuple")
        
        if len(key) == 2:
            # P[i, j] = seq
            i, j = key
            if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
                raise IndexError(f"index ({i}, {j}) out of bounds for shape {self.shape}")
            # if not isinstance(value, ComplexL0Sequence):
            #     raise TypeError(f"expected ComplexL0Sequence, got {type(value)}")
            self._sequences[(i, j)] = value
        
        elif len(key) == 3:
            # P[i, j, k] = c
            i, j, k = key
            if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
                raise IndexError(f"index ({i}, {j}) out of bounds for shape {self.shape}")
            self._ensure_sequence(i, j)[k] = value
        
        else:
            raise TypeError("indexing requires 2 or 3 indices")
    
    def __add__(self, other):
        """Addition of matrix sequences.
        
        Supports addition with:
        - Another ComplexL0MatrixSequence
        - A constant matrix (list of lists or array-like with .shape)
        
        Args:
            other: A ComplexL0MatrixSequence or a matrix of compatible shape
        
        Returns:
            ComplexL0MatrixSequence: The sum
        """
        if isinstance(other, ComplexL0MatrixSequence):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch: {self.shape} vs {other.shape}")
            
            # Determine range of both sequences
            self_indices = set(self._sequences.keys())
            other_indices = set(other._sequences.keys())
            all_indices = self_indices | other_indices
            
            result = type(self)(self.shape)
            
            for i, j in all_indices:
                self_seq = self._sequences.get((i, j))
                other_seq = other._sequences.get((i, j))
                
                if self_seq is None:
                    result._sequences[(i, j)] = self._create_sequence(other_seq.coeffs[:], other_seq.support_start)
                elif other_seq is None:
                    result._sequences[(i, j)] = self._create_sequence(self_seq.coeffs[:], self_seq.support_start)
                else:
                    # Both sequences exist, add them
                    result._sequences[(i, j)] = self_seq + other_seq
            
            return result
        
        else:
            # Try to interpret as a constant matrix
            rows, cols = _infer_shape(other)
            if (rows, cols) != self.shape:
                raise ValueError(f"matrix shape mismatch: {self.shape} vs {(rows, cols)}")
            
            result = type(self)(self.shape)
            
            # Copy all sequences from self
            for (i, j), seq in self._sequences.items():
                result._sequences[(i, j)] = self._create_sequence(seq.coeffs[:], seq.support_start)
            
            # Add constant matrix to the 0-th coefficient
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    # Get the value from the constant matrix
                    const_val = other[i][j] if isinstance(other, list) else other[i, j]
                    # Add to the 0-th coefficient
                    result._ensure_sequence(i, j)[0] = result[i, j][0] + const_val
            
            return result
    
    def __radd__(self, other):
        """Right addition (for scalar + matrix)."""
        return self + other
    
    def __neg__(self):
        """Negation of the matrix sequence.
        
        Returns:
            ComplexL0MatrixSequence: The negated matrix sequence
        """
        result = type(self)(self.shape)
        
        for (i, j), seq in self._sequences.items():
            result._sequences[(i, j)] = -seq
        
        return result
    
    def __sub__(self, other):
        """Subtraction of matrix sequences."""
        return self + (-other)
    
    def __rsub__(self, other):
        """Right subtraction (for constant matrix - matrix)."""
        return (-self) + other
    
    def _create_sequence(self, coeffs, support_start):
        """Factory method to create a sequence of the correct type.
        Override in subclasses to use Polynomial instead."""
        return ComplexL0Sequence(coeffs, support_start)
    

class MatrixPolynomial(ComplexL0MatrixSequence):
    """Represents a general matrix Laurent polynomial of one complex variable.

    Each entry (i, j) is a Polynomial (Laurent polynomial in z).
    
    Attributes:
        shape (tuple): (rows, cols) dimensions of the matrix.
        _sequences (dict): Dictionary mapping (i, j) -> Polynomial (polynomial coefficients).
    """

    def __init__(self, shape: tuple[int, int]):
        """Initializes a matrix polynomial.

        Args:
            shape: Tuple (rows, cols) specifying the matrix dimensions.
        """
        super().__init__(shape)

    def _create_sequence(self, coeffs, support_start):
        """Override to create Polynomial objects instead of ComplexL0Sequence."""
        return Polynomial(coeffs, support_start)

    def _ensure_sequence(self, i: int, j: int) -> Polynomial:
        """Get or create the polynomial at position (i, j)."""
        if (i, j) not in self._sequences:
            self._sequences[(i, j)] = Polynomial([])
        return self._sequences[(i, j)]

    def duplicate(self):
        """Creates a duplicate of the current matrix polynomial.

        Returns:
            MatrixPolynomial: A new instance with the same polynomials.
        """
        result = MatrixPolynomial(self.shape)
        for (i, j), poly in self._sequences.items():
            result._sequences[(i, j)] = Polynomial(poly.coeffs[:], poly.support_start)
        return result
    
    def shift(self, k: int):
        """Creates a new matrix polynomial equal to the current one, multiplied by `z^k`.
        
        Args:
            k (int): The power of z to multiply by.
        
        Returns:
            MatrixPolynomial: A new shifted matrix polynomial.
        """
        result = MatrixPolynomial(self.shape)
        for (i, j), poly in self._sequences.items():
            result._sequences[(i, j)] = Polynomial(poly.coeffs[:], poly.support_start + k)
        return result

    def effective_degree(self) -> int:
        """Returns the maximum effective degree across all polynomials.

        Returns:
            int: The largest (max_degree - min_degree) among all polynomials.
        """
        if not self._sequences:
            return -1
        return max(len(poly.coeffs) - 1 for poly in self._sequences.values())

    def conjugate(self):
        r"""Returns the conjugate matrix polynomial on the unit circle.

        Returns:
            MatrixPolynomial: The conjugate matrix polynomial.
        """
        result = MatrixPolynomial(self.shape)
        for (i, j), poly in self._sequences.items():
            result._sequences[(i, j)] = poly.conjugate()
        return result
    
    def sharp(self):
        r"""Conjugate with support adjusted.

        Returns:
            MatrixPolynomial: The sharp-conjugate matrix polynomial.
        """
        result = MatrixPolynomial(self.shape)
        for (i, j), poly in self._sequences.items():
            result._sequences[(i, j)] = poly.sharp()
        return result

    def schwarz_transform(self):
        r"""Returns the anti-analytic matrix polynomial.

        Returns:
            MatrixPolynomial: The Schwarz transform.
        """
        result = MatrixPolynomial(self.shape)
        for (i, j), poly in self._sequences.items():
            result._sequences[(i, j)] = poly.schwarz_transform()
        return result

    def __mul__(self, other):
        """Matrix polynomial multiplication.
        
        Supports:
        - Multiplication with a scalar
        - Multiplication with another MatrixPolynomial (matrix multiplication)
        
        Args:
            other: A scalar or MatrixPolynomial
        
        Returns:
            MatrixPolynomial: The product.
        """
        if isinstance(other, Number):
            result = MatrixPolynomial(self.shape)
            for (i, j), poly in self._sequences.items():
                result._sequences[(i, j)] = poly * other
            return result
        
        elif isinstance(other, MatrixPolynomial):
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"matrix shape mismatch for multiplication: {self.shape} x {other.shape}")
            
            result = MatrixPolynomial((self.shape[0], other.shape[1]))
            
            # (i, k) = sum over j of (i, j) * (j, k)
            for i in range(self.shape[0]):
                for k in range(other.shape[1]):
                    product_poly = None
                    for j in range(self.shape[1]):
                        self_poly = self._sequences.get((i, j))
                        other_poly = other._sequences.get((j, k))
                        
                        if self_poly is not None and other_poly is not None:
                            term = self_poly * other_poly
                            
                            if product_poly is None:
                                product_poly = term
                            else:
                                product_poly = product_poly + term
                    
                    if product_poly is not None:
                        result._sequences[(i, k)] = product_poly
            
            return result
        
        else:
            raise TypeError("MatrixPolynomial multiplication requires a scalar or another MatrixPolynomial.")
    
    def __rmul__(self, other):
        """Right multiplication (for scalar * matrix)."""
        return self * other
    
    def __truediv__(self, other):
        """Division by a scalar.
        
        Args:
            other: A scalar.
        
        Returns:
            MatrixPolynomial: The result.
        """
        if isinstance(other, Number):
            result = MatrixPolynomial(self.shape)
            for (i, j), poly in self._sequences.items():
                result._sequences[(i, j)] = poly / other
            return result
        
        raise TypeError("MatrixPolynomial division is only possible with scalars.")

    def __call__(self, z) -> list[list[generic_complex]]:
        """Evaluates the matrix polynomial using Horner's method.

        Args:
            z (complex): The point at which to evaluate.

        Returns:
            list[list[complex]]: The evaluated matrix.
        """
        result = [[bd.make_complex(0) for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        
        for (i, j), poly in self._sequences.items():
            result[i][j] = poly(z)
        
        return result

    def eval_at_roots_of_unity(self, N: int) -> list[list[list[generic_complex]]]:
        """Evaluates the matrix polynomial at the N-th roots of unity.

        Uses each entry's Polynomial.eval_at_roots_of_unity for efficient FFT-based evaluation.

        Args:
            N (int): Number of roots (will round up to next power of two).

        Returns:
            list[list[list[complex]]]: List of matrices evaluated at the roots.
        """
        N = next_power_of_two(N)

        # pre-allocate N matrices filled with zeros
        results = [
            [[bd.make_complex(0) for _ in range(self.shape[1])] for _ in range(self.shape[0])]
            for _ in range(N)
        ]

        # evaluate each polynomial entry at the roots (using its own fast routine)
        for (i, j), poly in self._sequences.items():
            vals = poly.eval_at_roots_of_unity(N)
            for k, v in enumerate(vals):
                results[k][i][j] = v

        return results
    
    def sup_norm(self, N=1024):
        """Estimates the supremum norm of the matrix polynomial over the unit circle.
        
        The norm is the maximum spectral norm across all sample points.
        
        Args:
            N (int, optional): The number of samples. Defaults to 1024.

        Returns:
            float: An estimate for the supremum norm.
        """
        evals = self.eval_at_roots_of_unity(N)
        max_norm = 0
        for mat in evals:
            # Compute spectral norm (largest singular value)
            # For simplicity, use Frobenius norm as approximation
            frob_norm = bd.sqrt(sum(abs(mat[i][j])**2 for i in range(self.shape[0]) for j in range(self.shape[1])))
            max_norm = max(max_norm, frob_norm)
        return max_norm
    
    def truncate(self, m: int, n: int):
        """Keeps only coefficients in degrees [m, n].

        Args:
            m (int): Lower bound of degree.
            n (int): Upper bound of degree.

        Returns:
            MatrixPolynomial: A new, truncated matrix polynomial.
        """
        result = MatrixPolynomial(self.shape)
        for (i, j), poly in self._sequences.items():
            result._sequences[(i, j)] = poly.truncate(m, n)
        return result
    
    def only_positive_degrees(self):
        """Discards all negative degrees, keeping only non-negative ones.
        
        Returns:
            MatrixPolynomial: A new polynomial with only non-negative degrees.
        """
        result = MatrixPolynomial(self.shape)
        for (i, j), poly in self._sequences.items():
            result._sequences[(i, j)] = poly.only_positive_degrees()
        return result

    def __str__(self):
        """String representation of the matrix polynomial.

        Returns:
            str: A description of the matrix polynomial.
        """
        return f"MatrixPolynomial{self.shape}"