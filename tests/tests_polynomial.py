
import unittest

import nlft_qsp.numerics as bd

from nlft_qsp.poly import ChebyshevTExpansion, Polynomial
from nlft_qsp.rand import random_sequence
from nlft_qsp.nlft import NonLinearFourierSequence


class PolynomialTestCase(unittest.TestCase):

    def test_get_set(self):
        p = Polynomial([1, 2, 3, 4], support_start=-3)

        self.assertEqual(p[-3], 1)
        self.assertEqual(p[-4], 0)
        self.assertEqual(p[-1], 3)
        self.assertEqual(p.effective_degree(), 3)

        p[-2] = 7
        self.assertEqual(p[-2], 7)
        self.assertEqual(p[-1], 3)
        self.assertEqual(p.effective_degree(), 3)

        p[1] = 9
        p[5] = 10
        self.assertEqual(p[1], 9)
        self.assertEqual(p[5], 10)
        self.assertEqual(p[3], 0)
        self.assertEqual(p.effective_degree(), 8)

        p[-5] = 20
        self.assertEqual(p[-5], 20)
        self.assertEqual(p[5], 10)
        self.assertEqual(p[-4], 0)
        self.assertEqual(p.effective_degree(), 10)

    def test_conjugate(self):
        p = Polynomial(random_sequence(10, 6), support_start=2)

        q = p.conjugate()

        for k in range(10):
            self.assertEqual(q[k], bd.conj(p[-k]))

    def test_truncate(self):
        p = Polynomial(range(20), support_start=-10)

        q = p.truncate(-5, 5)

        for k in p.support():
            if k not in q.support():
                self.assertEqual(q[k], 0)
            else:
                self.assertEqual(q[k], p[k])

    def test_add(self):
        p = bd.make_complex(1) + Polynomial([1, 2, 3])
        self.assertEqual(p.coeffs, [2, 2, 3])
        self.assertEqual(p.support_start, 0)

        p = Polynomial([1, 2, 3], support_start=2) - 4
        self.assertEqual(p.coeffs, [-4, 0, 1, 2, 3])
        self.assertEqual(p.support_start, 0)

        p = Polynomial([1, 2, 3], support_start=-3) + 7
        self.assertEqual(p.coeffs, [1, 2, 3, 7])
        self.assertEqual(p.support_start, -3)


        p = Polynomial([1, 2, 3], support_start=-3)
        q = p - Polynomial([4, 5, 6], support_start=-1)
        self.assertEqual(q.coeffs, [1, 2, -1, -5, -6])
        self.assertEqual(q.support_start, -3)

        q = p + Polynomial([4, 5, 6], support_start=-3)
        self.assertEqual(q.coeffs, [5, 7, 9])
        self.assertEqual(q.support_start, -3)

        q = p + Polynomial([4, 5, 6], support_start=-4)
        self.assertEqual(q.coeffs, [4, 6, 8, 3])
        self.assertEqual(q.support_start, -4)

    @bd.workdps(30)
    def test_mul(self):
        p = 3 * Polynomial([1, 2, 3], support_start=-10)
        self.assertEqual(p.coeffs, [3, 6, 9])
        self.assertEqual(p.support_start, -10)

        p = Polynomial([1, 2, 3], support_start=-1)
        q = p * Polynomial([5, 6, 7, 8], support_start=-5)

        for a, b in zip(q.coeffs, [5, 16, 34, 40, 37, 24]):
            self.assertAlmostEqual(a, b, delta=1e-25)
        self.assertEqual(q.support_start, -6)


    @bd.workdps(30)
    def test_call(self):
        p = Polynomial([3, 2, 1], support_start=0)
        q = Polynomial([5, -1, 0, 2], support_start=-3)

        self.assertAlmostEqual(p(1+1j), 5+4j, delta=10 * bd.machine_threshold())
        self.assertAlmostEqual(q(1+1j), 0.75-0.75j, delta=10 * bd.machine_threshold())

    @bd.workdps(30)
    def test_eval_at_roots_of_unity(self):
        seq = random_sequence(160000, 15)

        p = Polynomial(seq, support_start=0)
        q = Polynomial(seq, support_start=2)

        ep = p.eval_at_roots_of_unity(16)
        eq = q.eval_at_roots_of_unity(16)

        cep = [p(z) for z in bd.unitroots(16)]
        ceq = [q(z) for z in bd.unitroots(16)]
        for p1, p2 in zip(ep, cep):
            self.assertAlmostEqual(p1, p2, delta=10 * bd.machine_threshold())

        for z, q1, q2 in zip(bd.unitroots(16), eq, ceq):
            self.assertAlmostEqual(q1, q2, delta=10 * bd.machine_threshold())

        for z, a, b in zip(bd.unitroots(16), ep, eq):
            self.assertAlmostEqual(a * (z ** 2), b, delta=10 * bd.machine_threshold())


        ep = p.eval_at_roots_of_unity(8)
        eq = q.eval_at_roots_of_unity(8)

        cep = [p(z) for z in bd.unitroots(8)]
        ceq = [q(z) for z in bd.unitroots(8)]
        for p1, p2 in zip(ep, cep):
            self.assertAlmostEqual(p1, p2, delta=10 * bd.machine_threshold())

        for q1, q2 in zip(eq, ceq):
            self.assertAlmostEqual(q1, q2, delta=10 * bd.machine_threshold())

        for z, a, b in zip(bd.unitroots(8), ep, eq):
            self.assertAlmostEqual(a * (z ** 2), b, delta=10 * bd.machine_threshold())

    def test_schwarz_transform(self):
        p = Polynomial([])
        q = p.schwarz_transform()
        self.assertEqual(q.coeffs, [])

        p = Polynomial([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], support_start=-5)
        q = p.schwarz_transform()

        for k in p.support():
            if k < 0:
                self.assertEqual(q[k], 2*p[k])
            elif k == 0:
                self.assertEqual(q[k], p[k])
            else:
                self.assertEqual(q[k], 0)

    def test_chebyshev_expansion_to_laurent(self):
        T = ChebyshevTExpansion(random_sequence(10, 10))
        P = T.to_laurent()

        for k in range(32):
            theta = (k * bd.pi()) / 32
            z = bd.exp(1j * theta)
            x = bd.cos(theta)

            self.assertAlmostEqual(P(z), T(x), delta=bd.machine_threshold())

    def test_laurent_to_chebyshev_expansion(self):
        seq = random_sequence(10, 10)
        P = Polynomial(list(reversed(seq)) + seq[1:], support_start=-9)
        T = ChebyshevTExpansion(P)

        for k in range(32):
            theta = (k * bd.pi()) / 32
            z = bd.exp(1j * theta)
            x = bd.cos(theta)

            self.assertAlmostEqual(P(z), T(x), delta=bd.machine_threshold())


class NLFTTestCase(unittest.TestCase):

    @bd.workdps(10)
    def test_transform(self):
        nlft = NonLinearFourierSequence([])
        a, b = nlft.transform()
        self.assertEqual((a - 1).l2_squared_norm(), 0)
        self.assertEqual(b.l2_squared_norm(), 0)

        nlft = NonLinearFourierSequence([1, 2, 3, 4], support_start=0)
        a, b = nlft.transform()

        self.assertEqual(a.effective_degree(), 3)
        self.assertEqual(b.effective_degree(), 3)

        self.assertEqual(a.support_start, -3)
        self.assertEqual(b.support_start, 0)

        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=10 * bd.machine_threshold())
        
        for ak, ahk in zip(a.coeffs, [-0.0970143, 0.315296, -0.485071, 0.0242536]):
            self.assertAlmostEqual(ak, ahk, delta=1e-5)

        for bk, bhk in zip(b.coeffs, [0.0242536, -0.388057, -0.703353, 0.0970143]):
            self.assertAlmostEqual(bk, bhk, delta=1e-5)


        nlft = NonLinearFourierSequence([2, 3, 4], support_start=1)
        a, b = nlft.transform()

        self.assertEqual(a.effective_degree(), 2)
        self.assertEqual(b.effective_degree(), 2)

        self.assertEqual(a.support_start, -2)
        self.assertEqual(b.support_start, 1)

        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=10 * bd.machine_threshold())

        for ak, ahk in zip(a.coeffs, [-0.274398, -0.617395, 0.0342997]):
            self.assertAlmostEqual(ak, ahk, delta=1e-5)

        for bk, bhk in zip(b.coeffs, [0.0685994, -0.720294, 0.137199]):
            self.assertAlmostEqual(bk, bhk, delta=1e-5)

    def test_cheb2poly(self):
        P = Polynomial(random_sequence(10, 16))
        T = ChebyshevTExpansion.from_polynomial(P)

        s = 100000
        for l in range(s):
            x = 2*l/s - 1
            self.assertAlmostEqual(P(x), T(x), delta=bd.machine_threshold())

    def test_poly2cheb(self):
        T = ChebyshevTExpansion(random_sequence(10, 16))
        P = T.to_polynomial()

        s = 100000
        for l in range(s):
            x = 2*l/s - 1
            self.assertAlmostEqual(P(x), T(x), delta=bd.machine_threshold())


if __name__ == '__main__':
    unittest.main()