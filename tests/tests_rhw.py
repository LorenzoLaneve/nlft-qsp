
import unittest

import numpy as np
import scipy as sp

from tests import random_polynomial

import nlft_qsp.numerics as bd
from nlft_qsp import riemann_hilbert, weiss


class RHWTestCase(unittest.TestCase):
    
    @bd.workdps(30)
    def test_rhw(self):
        b = random_polynomial(16, eta=0.7)
        a, c = weiss.ratio(b)
        
        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=100 * bd.machine_eps())

        self.assertAlmostEqual(max([bd.abs(c(z) - b(z)/a(z)) for z in bd.unitroots(512)]), 0, delta=100 * bd.machine_eps())
        self.assertEqual(c.support().stop, b.support().stop)

        Ap, Bp = riemann_hilbert.factorize(c, 10, normalize=True)
        self.assertAlmostEqual((Ap * Ap.conjugate() + Bp * Bp.conjugate() - 1).l2_norm(), 0, delta=100 * bd.machine_eps())

    @bd.workdps(30)
    def test_inlft_rhw(self):
        b = random_polynomial(16, eta=0.5)
        a, c = weiss.ratio(b)

        nlft = riemann_hilbert.inlft(b, c)
        a2, b2 = nlft.transform()

        self.assertAlmostEqual((a - a2).l2_norm(), 0, delta=100 * bd.machine_eps())
        self.assertAlmostEqual((b - b2).l2_norm(), 0, delta=100 * bd.machine_eps())

    @bd.workdps(30)
    def test_half_cholesky_ldl(self):
        c = random_polynomial(16, eta=1)
        e0 = [1] + [0] * 15

        L = riemann_hilbert.half_cholesky_ldl(e0, reversed(c.coeffs))
        L = np.matrix(L) # we force numpy/scipy for this test, but L is computed with bd.

        B = np.matrix(riemann_hilbert.toeplitz(c, 0))
        K = np.eye(16) + B @ B.H

        L2, _, perm = sp.linalg.ldl(K)
        L2 = L2[perm, :]

        self.assertAlmostEqual(np.linalg.norm(L - L2), 0, delta=1e-5)


    @bd.workdps(30)
    def test_inlft_rhw_hc(self):
        b = random_polynomial(16, eta=0.5)
        a, c = weiss.ratio(b)

        nlft = riemann_hilbert.inlft_hc(b, c)
        a2, b2 = nlft.transform()

        self.assertAlmostEqual((a - a2).l2_norm(), 0, delta=100 * bd.machine_eps())
        self.assertAlmostEqual((b - b2).l2_norm(), 0, delta=100 * bd.machine_eps())


if __name__ == '__main__':
    unittest.main()