
import unittest

import numpy as np
import scipy as sp

import nlft_qsp.numerics as bd

from nlft_qsp.approximate import laurent_approximation
from nlft_qsp.poly import Polynomial
from nlft_qsp.solvers import janashia_lagvilava, half_cholesky


def random_matrix_polynomial(shape, N, eta):
    M = Polynomial(np.random.random_sample((N, *shape)))

    s = M.sup_norm()
    if s > eta:
        M *= (1 - eta) / s
    return M

class JanashiaLagvilavaTestCase(unittest.TestCase):

    def test_matrix_laurent_approx(self):
        d = 5
        A = [np.random.random_sample((d, d)) + 1j*np.random.random_sample((d, d)) for _ in range(32)]

        P = laurent_approximation(A)

        for j, Pz in zip(range(32), P.eval_at_roots_of_unity(32)):
            self.assertTrue(np.allclose(A[j], Pz, rtol=bd.machine_threshold()))

    def test_pointwise_cholesky(self):
        d = 15
        n = 15
        eta = 0.3

        A = random_matrix_polynomial((d, d), n+1, eta=eta) # target spectral factor

        I = np.eye(d)
        P = -A @ A.conjugate() + I

        M = janashia_lagvilava.pointwise_cholesky(P, 256)

        self.assertAlmostEqual((P - M @ M.conjugate()).l2_norm(), 0, delta=bd.machine_threshold())

    def test_janashia_lagvilava(self):
        d1 = 30
        d2 = 50
        n = 30

        Zeta = Polynomial(np.random.random_sample((n+1, d1, d2)) + 1j*np.random.random_sample((n+1, d1, d2))).conjugate()

        U = janashia_lagvilava.solve_jl_system(Zeta)
        self.assertEqual(U.shape, (d1 + d2, d1 + d2))

        L = Polynomial.block_matrix([
            [Polynomial(np.array([np.eye(d1)])), Polynomial(np.array([np.zeros((d1, d2))]))],
            [Zeta, Polynomial(np.array([np.eye(d2)]))]
        ])

        R = L @ U
        self.assertAlmostEqual((R - R.analytic_part()).l2_norm(), 0, delta=bd.machine_threshold())

    
    def test_schur_ldl(self):
        N = 50
        d1 = 10
        d2 = 20

        p = (np.random.random_sample((N, d1, d2)) + 1j * np.random.random_sample((N, d1, d2))) / N

        G = np.zeros((N * d1, N * d2), dtype=complex)
        for i in range(N):
            for j in range(i + 1):  # Lower triangular block structure
                G[i * d1 : (i + 1) * d1, j * d2 : (j + 1) * d2] = p[i - j]

        # Compute K = I + G G^*
        K = np.eye(N * d1) + G @ G.conj().T

        L, D = half_cholesky.half_cholesky_matrix_ldl(p)
        D = sp.linalg.block_diag(*D)

        self.assertTrue(np.allclose(K, L @ D @ L.conj().T, rtol=bd.machine_threshold()))

    def test_linear_system_toeplitz_displacement(self):
        N = 50
        d1 = 40
        d2 = 45

        p = np.random.random_sample((N, d1, d2)) + 1j * np.random.random_sample((N, d1, d2))

        G = np.zeros((N * d1, N * d2), dtype=complex)
        for i in range(N):
            for j in range(i + 1):  # Lower triangular block structure
                G[i * d1 : (i + 1) * d1, j * d2 : (j + 1) * d2] = p[i - j]

        # Compute K = I + G G^*
        K = np.eye(N * d1) + G @ G.conj().T

        C = np.random.random_sample((N, d1, d2)) + 1j * np.random.random_sample((N, d1, d2))
        
        X = half_cholesky.solve_system_displacement_structure(p, C)

        X = X.reshape(N * d1, -1)
        C = C.reshape(N * d1, -1)

        self.assertTrue(np.allclose(K @ X, C))

    def test_linear_system_hankel_displacement(self):
        N = 50
        d1 = 48
        d2 = 45

        p = np.random.random_sample((N, d1, d2)) + 1j * np.random.random_sample((N, d1, d2))

        G = np.zeros((N * d1, N * d2), dtype=complex)
        for i in range(N):
            for j in range(N - i):  # Anti-upper triangular block structure
                G[i * d1 : (i + 1) * d1, j * d2 : (j + 1) * d2] = p[i + j]

        # Compute K = I + G G^*
        K = np.eye(N * d1) + G @ G.conj().T

        C = np.random.random_sample((N, d1, d2)) + 1j * np.random.random_sample((N, d1, d2))
        
        X = half_cholesky.solve_system_displacement_structure(p, C, mode='hankel')

        X = X.reshape(N * d1, -1)
        C = C.reshape(N * d1, -1)

        self.assertTrue(np.allclose(K @ X, C))
        


if __name__ == '__main__':
    unittest.main()