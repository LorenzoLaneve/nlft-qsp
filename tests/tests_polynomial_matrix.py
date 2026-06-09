
import unittest

import nlft_qsp.numerics as bd
import numpy as np

from nlft_qsp.poly import Polynomial
from nlft_qsp.rand import random_complex, random_sequence


class MatrixPolynomialTestCase(unittest.TestCase):

    def test_init(self):
        M = Polynomial(shape=(2, 3))
        self.assertEqual(M.shape, (2, 3))

    def test_get_set_element(self):
        M = Polynomial(shape=(2, 2))
        
        # Set element at (2, 0, 1) -> coefficient of z^2 at position (0, 1)
        M[2, 0, 1] = 5 + 3j
        self.assertEqual(M[2, 0, 1], 5 + 3j)
        
        # Get element that doesn't exist
        self.assertEqual(M[5, 1, 1], 0)
        
        # Set multiple elements in same sequence
        M[-1, 0, 1] = 2 + 1j
        self.assertEqual(M[-1, 0, 1], 2 + 1j)
        self.assertEqual(M[2, 0, 1], 5 + 3j)

    def test_get_matrix_at_degree(self):
        M = Polynomial(shape=(2, 2))
        
        M[0, 0, 0] = 1
        M[0, 0, 1] = 2
        M[0, 1, 0] = 3
        M[0, 1, 1] = 4
        
        mat = M[0]
        self.assertTrue(np.all(mat == np.array([[1, 2], [3, 4]])))
        
        # Degree with no entries
        mat = M[5]
        self.assertTrue(np.all(mat == np.array([[0, 0], [0, 0]])))

    def test_matrix_operations(self):
        deg = 5
        shape = (2, 2)

        M1 = Polynomial(np.random.random_sample((deg, *shape)))
        M2 = Polynomial(np.random.random_sample((deg, *shape)))

        M3 = M1 @ M2
        M4 = M1 * M2
        M5 = M1 + M2
        M6 = M1 - M2

        for _ in range(8):
            z = random_complex(3)

            self.assertTrue(np.allclose(M1(z) @ M2(z), M3(z), rtol=bd.machine_threshold()))
            self.assertTrue(np.allclose(M1(z) * M2(z), M4(z), rtol=bd.machine_threshold()))
            self.assertTrue(np.allclose(M1(z) + M2(z), M5(z), rtol=bd.machine_threshold()))
            self.assertTrue(np.allclose(M1(z) - M2(z), M6(z), rtol=bd.machine_threshold()))

    def test_add_constant_scalar(self):
        M = Polynomial(shape=(2, 2))
        M[1, 0, 0] = 5
        
        M2 = M + 3
        
        # Constant is added to 0-th coefficient
        self.assertTrue(np.all(M2[0] == 3))
        
        # Original polynomial part is preserved
        self.assertEqual(M2[1, 0, 0], 5)

    def test_add_constant_matrix(self):
        M = Polynomial(shape=(2, 2))
        M[0, 1, 0] = 5
        
        M2 = M + np.array([[1, 2], [3, 4]])
        
        # Constant is added to 0-th coefficient
        self.assertTrue(np.all(M2[0] == np.array([[1, 2], [8, 4]])))

    def test_negation(self):
        M = Polynomial(shape=(2, 2))
        M[0, 0, 0] = 2 + 1j
        M[1, 1, -1] = 3 - 2j
        
        M_neg = -M
        self.assertEqual(M_neg[0, 0, 0], -2 - 1j)
        self.assertEqual(M_neg[1, 1, -1], -3 + 2j)

    def test_scalar_multiplication(self):
        M = Polynomial(shape=(2, 2))
        M[0, 0, 0] = 2
        M[1, 1, 1] = 3
        
        M2 = M * 5
        self.assertEqual(M2[0, 0, 0], 10)
        self.assertEqual(M2[1, 1, 1], 15)
        
        M3 = 2 * M
        self.assertEqual(M3[0, 0, 0], 4)
        self.assertEqual(M3[1, 1, 1], 6)

    def test_scalar_division(self):
        M = Polynomial(shape=(2, 2))
        M[0, 0, 0] = 10
        M[1, 1, 1] = 6
        
        M2 = M / 2
        self.assertEqual(M2[0, 0, 0], 5)
        self.assertEqual(M2[1, 1, 1], 3)

    def test_matrix_call(self):
        M = Polynomial(shape=(2, 2))
        
        # M = [[1 + z, 0], [0, 2 - z]]
        M[0, 0, 0] = 1
        M[1, 0, 0] = 1
        M[0, 1, 1] = 2
        M[1, 1, 1] = -1

        # Expected: [[1 + (2+1j), 0], [0, 2 - (2+1j)]] = [[3+1j, 0], [0, -1j]]
        self.assertTrue(np.allclose(M(2 + 1j), np.array([[3+1j, 0], [0, -1j]]), rtol=bd.machine_threshold()))

    def test_eval_at_roots_of_unity(self):
        deg = 5
        shape = (2, 2)

        M = Polynomial(np.random.random_sample((deg, *shape)))

        N = 16
        ep = M.eval_at_roots_of_unity(N)
        cep = [M(z) for z in bd.unitroots(N)]

        # Check that each evaluation is a 2x2 matrix
        for mat in ep:
            self.assertEqual(len(mat), 2)
            self.assertEqual(len(mat[0]), 2)

        ep_np = [np.array(mat, dtype=complex) for mat in ep]
        cep_np = [np.array(mat, dtype=complex) for mat in cep]

        for _ in range(len(ep)):
            self.assertTrue(np.allclose(ep_np, cep_np, rtol=bd.machine_threshold()))

    def test_duplicate(self):
        M = Polynomial(shape=(2, 2))
        M[0, 0, 0] = 1 + 2j
        M[1, 1, -1] = 3
        
        M2 = M.duplicate()
        
        self.assertEqual(M2[0, 0, 0], 1 + 2j)
        self.assertEqual(M2[1, 1, -1], 3)
        
        # Modify original, duplicate should be unaffected
        M[0, 0, 0] = 5
        self.assertEqual(M2[0, 0, 0], 1 + 2j)

    def test_shift(self):
        M = Polynomial(shape=(2, 2))
        M[0, 0, 0] = 1
        M[1, 0, 0] = 2
        
        M2 = M.shift(3)
        
        # Degrees should shift by 3
        self.assertEqual(M2[3, 0, 0], 1)
        self.assertEqual(M2[4, 0, 0], 2)
        self.assertEqual(M2[1, 0, 0], 0)

        deg = 5
        shape = (2, 2)
        M1 = Polynomial(np.random.random_sample((deg, *shape)))
        M2 = M1.shift(5)
        for _ in range(8):
            z = random_complex(3)
            self.assertTrue(np.allclose(M1(z) * (z ** 5), M2(z), rtol=bd.machine_threshold()))

    def test_conjugate(self):
        deg = 5
        shape = (2, 2)
        M = Polynomial(np.random.random_sample((deg, *shape)))
        M_star = M.conjugate()

        for z in bd.unitroots(32): # check schwarz reflection
            self.assertTrue(np.allclose(M(1/np.conj(z)).T.conj(), M_star(z), rtol=bd.machine_threshold()))

    def test_truncate(self):
        """Test truncation of matrix polynomial."""
        M = Polynomial(shape=(2, 2))
        M[-2, 0, 0] = 1
        M[0, 0, 0] = 2
        M[2, 0, 0] = 3

        M2 = M.truncate(-1, 1)
        self.assertEqual(M2[-2, 0, 0], 0)
        self.assertEqual(M2[0, 0, 0], 2)
        self.assertEqual(M2[2, 0, 0], 0)

    def test_shape_validation(self):
        """Test shape validation."""
        M1 = Polynomial(shape=(2, 3))
        M2 = Polynomial(shape=(3, 2))
        
        # Should work: 2x3 @ 3x2
        M3 = M1 @ M2
        self.assertEqual(M3.shape, (2, 2))

        with self.assertRaises(ValueError):
            M1 @ M1

        with self.assertRaises(ValueError):
            M1 + M2

        with self.assertRaises(ValueError):
            M1 * M2

        


if __name__ == '__main__':
    unittest.main()