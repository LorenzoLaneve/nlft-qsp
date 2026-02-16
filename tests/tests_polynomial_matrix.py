
import unittest

import nlft_qsp.numerics as bd
import numpy as np

from nlft_qsp.poly import Polynomial
from nlft_qsp.poly_matrix import MatrixPolynomial
from nlft_qsp.rand import random_sequence
from nlft_qsp.nlft import NonLinearFourierSequence


class MatrixPolynomialTestCase(unittest.TestCase):

    def test_init(self):
        """Test MatrixPolynomial initialization."""
        M = MatrixPolynomial((2, 3))
        self.assertEqual(M.shape, (2, 3))
        self.assertEqual(len(M._sequences), 0)

    def test_get_set_sequence(self):
        """Test getting and setting entire sequences."""
        M = MatrixPolynomial((2, 2))
        
        # Get returns empty sequence initially
        seq = M[0, 1]
        self.assertEqual(len(seq.coeffs), 0)
        
        # Set a sequence
        p = Polynomial([1, 2, 3], support_start=-1)
        M[0, 1] = p
        
        self.assertEqual(M[0, 1].coeffs, [1, 2, 3])
        self.assertEqual(M[0, 1].support_start, -1)

    def test_get_set_element(self):
        """Test getting and setting individual coefficients."""
        M = MatrixPolynomial((2, 2))
        
        # Set element at (0, 1, 2) -> coefficient of z^2 at position (0, 1)
        M[0, 1, 2] = 5 + 3j
        self.assertEqual(M[0, 1, 2], 5 + 3j)
        
        # Get element that doesn't exist
        self.assertEqual(M[1, 1, 5], bd.make_complex(0))
        
        # Set multiple elements in same sequence
        M[0, 1, -1] = 2 + 1j
        self.assertEqual(M[0, 1, -1], 2 + 1j)
        self.assertEqual(M[0, 1, 2], 5 + 3j)

    def test_get_matrix_at_degree(self):
        """Test getting the full matrix at a specific degree."""
        M = MatrixPolynomial((2, 2))
        
        M[0, 0, 0] = 1
        M[0, 1, 0] = 2
        M[1, 0, 0] = 3
        M[1, 1, 0] = 4
        
        mat = M[:, :, 0]
        self.assertEqual(mat, [[1, 2], [3, 4]])
        
        # Degree with no entries
        mat = M[:, :, 5]
        self.assertEqual(mat, [[0, 0], [0, 0]])

    def test_add_matrix_sequences(self):
        """Test addition of two matrix polynomials."""
        M1 = MatrixPolynomial((2, 2))
        M2 = MatrixPolynomial((2, 2))
        
        M1[0, 0, 0] = 1
        M1[1, 1, 1] = 2
        
        M2[0, 0, 0] = 3
        M2[0, 1, 1] = 4
        
        M3 = M1 + M2
        self.assertEqual(M3[0, 0, 0], 4)
        self.assertEqual(M3[0, 1, 1], 4)
        self.assertEqual(M3[1, 1, 1], 2)

    def test_add_constant_matrix(self):
        """Test adding a constant matrix to matrix polynomial."""
        M = MatrixPolynomial((2, 2))
        M[0, 0, 1] = 5
        
        const = [[1, 2], [3, 4]]
        M2 = M + const
        
        # Constant is added to 0-th coefficient
        self.assertEqual(M2[0, 0, 0], 1)
        self.assertEqual(M2[0, 1, 0], 2)
        self.assertEqual(M2[1, 0, 0], 3)
        self.assertEqual(M2[1, 1, 0], 4)
        
        # Original polynomial part is preserved
        self.assertEqual(M2[0, 0, 1], 5)

    def test_negation(self):
        """Test negation of matrix polynomial."""
        M = MatrixPolynomial((2, 2))
        M[0, 0, 0] = 2 + 1j
        M[1, 1, -1] = 3 - 2j
        
        M_neg = -M
        self.assertEqual(M_neg[0, 0, 0], -2 - 1j)
        self.assertEqual(M_neg[1, 1, -1], -3 + 2j)

    def test_subtraction(self):
        """Test subtraction of matrix polynomials."""
        M1 = MatrixPolynomial((2, 2))
        M2 = MatrixPolynomial((2, 2))
        
        M1[0, 0, 0] = 5
        M2[0, 0, 0] = 2
        
        M3 = M1 - M2
        self.assertEqual(M3[0, 0, 0], 3)

    def test_scalar_multiplication(self):
        """Test multiplication by scalar."""
        M = MatrixPolynomial((2, 2))
        M[0, 0, 0] = 2
        M[1, 1, 1] = 3
        
        M2 = M * 5
        self.assertEqual(M2[0, 0, 0], 10)
        self.assertEqual(M2[1, 1, 1], 15)
        
        M3 = 2 * M
        self.assertEqual(M3[0, 0, 0], 4)
        self.assertEqual(M3[1, 1, 1], 6)

    def test_scalar_division(self):
        """Test division by scalar."""
        M = MatrixPolynomial((2, 2))
        M[0, 0, 0] = 10
        M[1, 1, 1] = 6
        
        M2 = M / 2
        self.assertEqual(M2[0, 0, 0], 5)
        self.assertEqual(M2[1, 1, 1], 3)

    @bd.workdps(30)
    def test_matrix_multiplication(self):
        """Test matrix polynomial multiplication via pointwise evaluation.

        Build two random MatrixPolynomial instances of degree 5 and check
        M1(z) @ M2(z) == (M1 * M2)(z) for several random z.
        """
        import random

        deg = 5
        shape = (2, 2)

        M1 = MatrixPolynomial(shape)
        M2 = MatrixPolynomial(shape)

        # populate with random polynomials (support_start = 0)
        for i in range(shape[0]):
            for j in range(shape[1]):
                M1[i, j] = Polynomial(random_sequence(1.0, deg + 1), support_start=0)
                M2[i, j] = Polynomial(random_sequence(1.0, deg + 1), support_start=0)

        M3 = M1 * M2

        trials = 8
        for _ in range(trials):
            z = complex(random.random() * 2 - 1, random.random() * 2 - 1)
            A = M1(z)
            B = M2(z)
            C = M3(z)

            # compute A @ B using numpy
            A_np = np.array(A, dtype=complex)
            B_np = np.array(B, dtype=complex)
            prod_np = A_np @ B_np
            C_np = np.array(C, dtype=complex)

            # compare entrywise
            tol = 10 * bd.machine_threshold()
            for i in range(shape[0]):
                for j in range(shape[1]):
                    self.assertAlmostEqual(prod_np[i, j], C_np[i, j], delta=tol)

    @bd.workdps(30)
    def test_matrix_call(self):
        """Test evaluating matrix polynomial at a point."""
        M = MatrixPolynomial((2, 2))
        
        # M = [[1 + z, 0], [0, 2 - z]]
        M[0, 0, 0] = 1
        M[0, 0, 1] = 1
        M[1, 1, 0] = 2
        M[1, 1, 1] = -1
        
        result = M(2 + 1j)
        
        # Expected: [[1 + (2+1j), 0], [0, 2 - (2+1j)]] = [[3+1j, 0], [0, -1j]]
        self.assertAlmostEqual(result[0][0], 3 + 1j, delta=1e-10)
        self.assertAlmostEqual(result[0][1], 0, delta=1e-10)
        self.assertAlmostEqual(result[1][0], 0, delta=1e-10)
        self.assertAlmostEqual(result[1][1], -1j, delta=1e-10)

    @bd.workdps(30)
    def test_eval_at_roots_of_unity(self):
        """Test evaluation at roots of unity."""
        deg = 5
        shape = (2, 2)

        M = MatrixPolynomial(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                M[i, j] = Polynomial(random_sequence(1.0, deg + 1), support_start=0)
                M[i, j] = Polynomial(random_sequence(1.0, deg + 1), support_start=0)

        N = 16
        ep = M.eval_at_roots_of_unity(N)
        cep = [M(z) for z in bd.unitroots(N)]

        # Check that each evaluation is a 2x2 matrix
        for mat in ep:
            self.assertEqual(len(mat), 2)
            self.assertEqual(len(mat[0]), 2)

        ep_np = [np.array(mat, dtype=complex) for mat in ep]
        cep_np = [np.array(mat, dtype=complex) for mat in cep]

        tol = 10 * bd.machine_threshold()
        for k in range(len(ep)):
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    self.assertAlmostEqual(ep_np[k][i][j], cep_np[k][i][j], delta=tol)

    def test_duplicate(self):
        """Test duplication of matrix polynomial."""
        M = MatrixPolynomial((2, 2))
        M[0, 0, 0] = 1 + 2j
        M[1, 1, -1] = 3
        
        M2 = M.duplicate()
        
        self.assertEqual(M2[0, 0, 0], 1 + 2j)
        self.assertEqual(M2[1, 1, -1], 3)
        
        # Modify original, duplicate should be unaffected
        M[0, 0, 0] = 5
        self.assertEqual(M2[0, 0, 0], 1 + 2j)

    def test_shift(self):
        """Test shifting matrix polynomial by power of z."""
        M = MatrixPolynomial((2, 2))
        M[0, 0, 0] = 1
        M[0, 0, 1] = 2
        
        M2 = M.shift(3)
        
        # Degrees should shift by 3
        self.assertEqual(M2[0, 0, 3], 1)
        self.assertEqual(M2[0, 0, 4], 2)
        self.assertEqual(M2[0, 0, 1], bd.make_complex(0))

    def test_conjugate(self):
        """Test conjugation of matrix polynomial."""
        M = MatrixPolynomial((2, 2))
        M[0, 0, 0] = 1 + 2j
        M[0, 0, 1] = 3 - 1j
        
        M_conj = M.conjugate()
        
        self.assertEqual(M_conj[0, 0, 0], 1 - 2j)
        self.assertEqual(M_conj[0, 0, -1], 3 + 1j)

    def test_truncate(self):
        """Test truncation of matrix polynomial."""
        M = MatrixPolynomial((2, 2))
        M[0, 0, -2] = 1
        M[0, 0, 0] = 2
        M[0, 0, 2] = 3
        
        M2 = M.truncate(-1, 1)
        
        self.assertEqual(M2[0, 0, -2], bd.make_complex(0))
        self.assertEqual(M2[0, 0, 0], 2)
        self.assertEqual(M2[0, 0, 2], bd.make_complex(0))

    def test_shape_validation(self):
        """Test shape validation."""
        M1 = MatrixPolynomial((2, 3))
        M2 = MatrixPolynomial((3, 2))
        
        # Should work: 2x3 * 3x2
        M3 = M1 * M2
        self.assertEqual(M3.shape, (2, 2))
        
        # Should fail: 2x3 + 3x2
        with self.assertRaises(ValueError):
            M1 + M2


if __name__ == '__main__':
    unittest.main()