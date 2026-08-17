
import unittest

import numpy as np
import scipy as sp

import nlft_qsp.numerics as bd
from nlft_qsp.rand import random_polynomial

from nlft_qsp.solvers import weiss, riemann_hilbert, nlfft, layer_stripping, half_cholesky
from nlft_qsp.util import unitroots


class RHWTestCase(unittest.TestCase):
    
    def test_rhw(self):
        b = random_polynomial(16, eta=0.5)
        a, c = weiss.ratio(b)
        
        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=bd.machine_threshold())

        self.assertAlmostEqual(max([np.abs(c(z) - b(z)/a(z)) for z in unitroots(512)]), 0, delta=bd.machine_threshold())
        self.assertEqual(c.support().stop, b.support().stop)

        Ap, Bp = riemann_hilbert.factorize(c, 10, normalize=True)
        self.assertAlmostEqual((Ap * Ap.conjugate() + Bp * Bp.conjugate() - 1).l2_norm(), 0, delta=bd.machine_threshold())

    def test_inlft_rhw(self):
        b = random_polynomial(16, eta=0.5)
        a, c = weiss.ratio(b)

        nlft = riemann_hilbert.inlft(b, c)
        a2, b2 = nlft.transform()

        self.assertAlmostEqual((a - a2).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((b - b2).l2_norm(), 0, delta=bd.machine_threshold())

    def test_half_cholesky_ldl(self):
        c = random_polynomial(16, eta=1)
        e0 = [1] + [0] * 15

        L = half_cholesky.half_cholesky_ldl(e0, reversed(c.coeffs))
        L = np.array(L, dtype=np.complex128)

        B = riemann_hilbert.toeplitz(c, 0)
        K = np.eye(16) + B @ B.conj().T
        for k in range(16): # just to suppress the annoying ComplexWarning
            K[k, k] = np.real(K[k, k])

        L2, _, perm = sp.linalg.ldl(K)
        L2 = L2[perm, :]

        self.assertAlmostEqual(np.linalg.norm(L - L2), 0, delta=1e-5)

    def test_inlft_rhw_hc(self):
        b = random_polynomial(16, eta=0.5)
        a, c = weiss.ratio(b)

        nlft = half_cholesky.inlft(b, c)
        a2, b2 = nlft.transform()

        self.assertAlmostEqual((a - a2).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((b - b2).l2_norm(), 0, delta=bd.machine_threshold())

    def test_inlft_nlfft(self):
        b = random_polynomial(1600, eta=0.5)
        a = weiss.complete(b)

        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=bd.machine_threshold())

        nlft = nlfft.inlft(a, b)
        a2, b2 = nlft.transform()

        self.assertAlmostEqual((a - a2).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((b - b2).l2_norm(), 0, delta=bd.machine_threshold())

    def test_inlft_layer_stripping(self):
        b = random_polynomial(1600, eta=0.5)
        a = weiss.complete(b)

        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=bd.machine_threshold())

        nlft = layer_stripping.inlft(a, b)
        a2, b2 = nlft.transform()

        self.assertAlmostEqual((a - a2).l2_norm(), 0, delta=bd.machine_threshold())
        self.assertAlmostEqual((b - b2).l2_norm(), 0, delta=bd.machine_threshold())


if __name__ == '__main__':
    unittest.main()