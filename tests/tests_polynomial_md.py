
from math import sqrt
import unittest

import nlft_qsp.numerics as bd

from nlft_qsp.nlft_md import StairlikeSequence2D
from nlft_qsp.poly_md import PolynomialMD
from nlft_qsp.rand import random_complex, random_list, random_sequence, random_stairlike_sequence_2d
from nlft_qsp.poly import Polynomial
from nlft_qsp.solvers import weiss_md


class PolynomialMDTestCase(unittest.TestCase):

    def test_get_set(self):
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]], support_start=(0, 0))
        
        self.assertEqual(p[0,0], 1)
        self.assertEqual(p[0,1], 2)
        self.assertEqual(p[0,2], 3)
        self.assertEqual(p[1,0], 4)
        self.assertEqual(p[1,1], 5)
        self.assertEqual(p[1,2], 6)
        self.assertEqual(p[2,0], 7)
        self.assertEqual(p[2,1], 8)
        self.assertEqual(p[2,2], 9)

        self.assertEqual(p[-2,7], 0)
        p[-2,7] = 15
        self.assertEqual(p[-2,7], 15)
        self.assertEqual(p[-3,5], 0)
        self.assertEqual(p[-1,7], 0)

        self.assertEqual(p[1,8], 0)
        p[1,8] = 72
        self.assertEqual(p[1,8], 72)
        
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]], support_start=(3, 2))
        self.assertEqual(p[3, 2], 1)

    def test_support(self):
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(0, 0))
        
        r1, r2 = p.support()
        self.assertEqual(r1, range(0, 2))
        self.assertEqual(r2, range(0, 3))

        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(0, 2))
        
        r1, r2 = p.support()
        self.assertEqual(r1, range(0, 2))
        self.assertEqual(r2, range(2, 5))

        p = PolynomialMD(
            [[[1, 2], [3, 4], [5, 6]],
             [[1, 2], [3, 4], [5, 6]]], support_start=(1, 4, 5)
        )
        r1, r2, r3 = p.support()
        self.assertEqual(r1, range(1, 3))
        self.assertEqual(r2, range(4, 7))
        self.assertEqual(r3, range(5, 7))

    def test_coeff_list(self):
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(0, 0))
        self.assertEqual(p.coeff_list(), [[1, 2, 3], [4, 5, 6]])

        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(2, 3))
        self.assertEqual(p.coeff_list(), [[1, 2, 3], [4, 5, 6]])

    def test_support_start(self):
        p = PolynomialMD(
            [[[1, 2], [3, 4], [5, 6]],
             [[1, 2], [3, 4], [5, 6]]], support_start=(0, 0, 0)
        )
        self.assertEqual(p.support_start, (0, 0, 0))

        p[-2, 0, 0] = 3
        self.assertEqual(p.support_start, (-2, 0, 0))

        p[-5, -3, -2] = 5
        self.assertEqual(p.support_start, (-5, -3, -2))

    def test_effective_degree(self):
        p = PolynomialMD(
            [[[1, 2], [3, 4], [5, 6]],
             [[1, 2], [3, 4], [5, 6]]], support_start=(0, 0, 0)
        )

        self.assertEqual(p.effective_degree(), (1, 2, 1))

        p[0, 0, 5] = 7
        self.assertEqual(p.effective_degree(), (1, 2, 5))

    def test_conjugate(self):
        p = PolynomialMD([[[0]]], (0, 0, 0))
        p[1, 2, 3] = 72 + 7j
        p[-5, -3, 1] = 36 - 5j

        self.assertEqual(p[1, 2, 3], 72 + 7j)
        self.assertEqual(p[-5, -3, 1], 36 - 5j)
        q = p.conjugate()
        self.assertEqual(q[-1, -2, -3], 72 - 7j)
        self.assertEqual(q[5, 3, -1], 36 + 5j)

    def test_add(self):
        p = PolynomialMD(
            [[1, 2, 3],
             [4, 5, 6]], support_start=(0, 0))
        
        q = PolynomialMD(
            [[7, 8, 9],
             [1, 3, 6]], support_start=(1, 1))
        
        self.assertEqual((p+q).coeff_list(),
                         [[1,   2,   3, 0],
                          [4, 5+7, 6+8, 9],
                          [0, 1,     3, 6]])
        self.assertAlmostEqual(((p+q) - (q+p)).l2_norm(), 0, delta=bd.machine_threshold())
        
        self.assertEqual((p-q).coeff_list(),
                         [[1,   2,   3,  0],
                          [4, 5-7, 6-8, -9],
                          [0,  -1,  -3, -6]])
        self.assertAlmostEqual(((p-q) + (q-p)).l2_norm(), 0, delta=bd.machine_threshold())
        
        self.assertEqual((q+35).coeff_list(),
                         [[35, 0, 0, 0],
                          [0,  7, 8, 9],
                          [0,  1, 3, 6]])
        
    def test_mul(self):
        p = PolynomialMD([[0, 1], [1, 0]], support_start=(0,0)) # x + y
        q = PolynomialMD([[0, 1], [-1, 0]], support_start=(1,0)) # x^2 - xy

        r = p*q
        self.assertEqual(r.support_start, (1,0))
        self.assertEqual(r.coeff_list(), [
            [0, 0, 1],
            [0, 0, 0],
            [-1, 0, 0]
        ])

        p = PolynomialMD([[0, 1], [1, 7]], support_start=(0,3)) # y^3 (x + y + 7xy)
        q = PolynomialMD([[0, 1], [-1, 0]], support_start=(1,0)) # x^2 - xy
        # x y^3 (x + y + 7xy) (x - y) = x y^3 (x^2 - y^2 + 7x^2 y - 7xy^2)
        
        r = p*q
        self.assertEqual(r.support_start, (1,3))
        self.assertEqual(r.coeff_list(), [
            [0, 0, 1],
            [0, 0, 7],
            [-1, -7, 0]
        ])

    def test_eval(self):
        p = PolynomialMD([ # P(x, y) = 2x^2 + 3xy + y^2
            [0, 0, 1],
            [0, 3, 0],
            [2, 0, 0]
        ], (0,0))
        self.assertEqual(p(1, 2), 12)

        p = PolynomialMD([ # P(x, y) = y^2(2x^2 + 3xy + y^2)
            [0, 0, 1],
            [0, 3, 0],
            [2, 0, 0]
        ], (0,2))
        self.assertEqual(p(1, 2), 48)

        p = PolynomialMD([ # P(x, y, z) = x + yz - z^2
            [[0, 0, -1],
             [0, 1, 0]],
            [[1, 0, 0],
             [0, 0, 0]]
        ], (0,0,0))
        self.assertEqual(p(2, 3, 1), 4)

    def test_eval_at_roots_of_unity(self):
        seq = random_sequence(10000, (4, 4, 4))

        p = PolynomialMD(seq, (0, 0, 0))
        q = PolynomialMD(seq, (0, 2, 1)) # q = y^2 z p
        
        ep = p.eval_at_roots_of_unity((6, 7, 9))
        eq = q.eval_at_roots_of_unity((6, 7, 9))

        for i, x in zip(range(8), bd.unitroots(8)):
            for j, y in zip(range(8), bd.unitroots(8)):
                for k, z in zip(range(16), bd.unitroots(16)):
                    self.assertAlmostEqual(ep[i][j][k], p(x, y, z), delta=10 * bd.machine_threshold())
                    self.assertAlmostEqual(eq[i][j][k], q(x, y, z), delta=10 * bd.machine_threshold())
                    self.assertAlmostEqual(eq[i][j][k], (y ** 2) * z * ep[i][j][k], delta=10 * bd.machine_threshold())

class Polynomial1DToMDTestCase(unittest.TestCase):

    def test_eval(self):
        seq = random_sequence(10, 16)
        w = random_complex(10)
        y = random_complex(10)

        p1 = PolynomialMD([seq], support_start=(0,0))
        p2 = PolynomialMD([[c] for c in seq], support_start=(0,0))
        p3 = PolynomialMD([[seq]], support_start=(0,0,0))
        q = Polynomial(seq, support_start=0)

        for z in bd.unitroots(16):
            self.assertAlmostEqual(p1(w, z), q(z), delta=10*bd.machine_threshold())
            self.assertAlmostEqual(p2(z, w), q(z), delta=10*bd.machine_threshold())
            self.assertAlmostEqual(p3(y, w, z), q(z), delta=10*bd.machine_threshold())

        p1 = PolynomialMD([seq], support_start=(0,3))
        p2 = PolynomialMD([[c] for c in seq], support_start=(3,0))
        p3 = PolynomialMD([[seq]], support_start=(0,0,3))
        q = Polynomial(seq, support_start=3)

        for z in bd.unitroots(16):
            self.assertAlmostEqual(p1(w, z), q(z), delta=10*bd.machine_threshold())
            self.assertAlmostEqual(p2(z, w), q(z), delta=10*bd.machine_threshold())
            self.assertAlmostEqual(p3(y, w, z), q(z), delta=10*bd.machine_threshold())

        e1 = p1.eval_at_roots_of_unity(16)
        e2 = p2.eval_at_roots_of_unity(16)
        e3 = p3.eval_at_roots_of_unity(16)

        eq = q.eval_at_roots_of_unity(16)

        for k in range(16):

            for h in range(16):
                self.assertAlmostEqual(e1[h][k], eq[k], delta=10*bd.machine_threshold())
                self.assertAlmostEqual(e2[k][h], eq[k], delta=10*bd.machine_threshold())

                for j in range(16):
                    self.assertAlmostEqual(e3[j][h][k], eq[k], delta=10*bd.machine_threshold())

    def test_eval_at_roots_of_unity(self):
        seq = random_sequence(10, 16)

        e1 = PolynomialMD([seq], support_start=(0,3))               .eval_at_roots_of_unity(16)
        e2 = PolynomialMD([[c] for c in seq], support_start=(3,0))  .eval_at_roots_of_unity(16)
        e3 = PolynomialMD([[seq]], support_start=(0,0,3))           .eval_at_roots_of_unity(16)

        eq = Polynomial(seq, support_start=3).eval_at_roots_of_unity(16)

        for k in range(16):
            for h in range(16):
                self.assertAlmostEqual(e1[h][k], eq[k], delta=10*bd.machine_threshold())
                self.assertAlmostEqual(e2[k][h], eq[k], delta=10*bd.machine_threshold())

                for j in range(16):
                    self.assertAlmostEqual(e3[j][h][k], eq[k], delta=10*bd.machine_threshold())

    def test_add(self):
        seq1 = random_sequence(10, 16)
        seq2 = random_sequence(10, 16)

        p1_md = PolynomialMD([seq1], support_start=(0,0))
        q1_md = PolynomialMD([seq2], support_start=(0,-5))

        p2_md = PolynomialMD([[c] for c in seq1], support_start=(0,0))
        q2_md = PolynomialMD([[c] for c in seq2], support_start=(-5,0))

        p = Polynomial(seq1, support_start=0)
        q = Polynomial(seq2, support_start=-5)

        r1_md = p1_md + q1_md
        r2_md = p2_md + q2_md
        r = p + q

        w = random_complex(10)
        for z in bd.unitroots(1024):
            self.assertAlmostEqual(r1_md(w, z), r(z), delta=10*bd.machine_threshold())
            self.assertAlmostEqual(r2_md(z, w), r(z), delta=10*bd.machine_threshold())

    def test_mul(self):
        seq1 = random_sequence(10, 16)
        seq2 = random_sequence(10, 16)

        p1_md = PolynomialMD([seq1], support_start=(0,0))
        q1_md = PolynomialMD([seq2], support_start=(0,-5))

        p2_md = PolynomialMD([[c] for c in seq1], support_start=(0,0))
        q2_md = PolynomialMD([[c] for c in seq2], support_start=(-5,0))

        p = Polynomial(seq1, support_start=0)
        q = Polynomial(seq2, support_start=-5)

        r1_md = p1_md * q1_md
        r2_md = p2_md * q2_md
        r = p * q

        w = random_complex(10)
        for z in bd.unitroots(1024):
            self.assertAlmostEqual(r1_md(w, z), r(z), delta=10*bd.machine_threshold())
            self.assertAlmostEqual(r2_md(z, w), r(z), delta=10*bd.machine_threshold())

    def test_schwarz_transform(self):
        seq = random_sequence(10, 10)
        w = random_complex(10)
        y = random_complex(10)

        p1 = PolynomialMD([seq], support_start=(0,0))               .schwarz_transform()
        p2 = PolynomialMD([[c] for c in seq], support_start=(0,0))  .schwarz_transform()
        p3 = PolynomialMD([[seq]], support_start=(0,0,0))           .schwarz_transform()
        q = Polynomial(seq, support_start=0)                        .schwarz_transform()

        for z in bd.unitroots(16):
            self.assertAlmostEqual(p1(w, z), q(z), delta=10*bd.machine_threshold())
            self.assertAlmostEqual(p2(z, w), q(z), delta=10*bd.machine_threshold())
            self.assertAlmostEqual(p3(y, w, z), q(z), delta=10*bd.machine_threshold())



class StairlikeSequence2DTestCase(unittest.TestCase):

    def test_init_get(self):
        s = StairlikeSequence2D([[1, 2], [3], [4, 5, 6], [7, 8]])

        self.assertEqual(s.support_x(), range(0, 4))
        self.assertEqual(s.support_y(), range(0, 5))

        self.assertEqual(s[0,0], 1)
        self.assertEqual(s[0,1], 2)
        self.assertEqual(s[1,1], 3)
        self.assertEqual(s[2,1], 4)
        self.assertEqual(s[2,2], 5)
        self.assertEqual(s[2,3], 6)
        self.assertEqual(s[3,3], 7)
        self.assertEqual(s[3,4], 8)

        self.assertEqual(s[1,0], 0)
        self.assertEqual(s[-1,0], 0)

    def test_nlft_transform(self):
        nlft = StairlikeSequence2D([[1, 2], [3], [4]])
        a, b = nlft.transform()

        self.assertAlmostEqual(a[0,0],     1/(10*sqrt(17)),  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(a[0,-1],    -1/(5*sqrt(17)),  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(a[-1,0],    -9/(5*sqrt(17)),  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(a[-1,-1],   21/(10*sqrt(17)), delta=10*bd.machine_threshold())
        self.assertAlmostEqual(a[-2,0],    -4/(5*sqrt(17)),  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(a[-2,-1],   -2/(5*sqrt(17)),  delta=10*bd.machine_threshold())

        self.assertAlmostEqual(b[0,0],     1/(10*sqrt(17)),   delta=10*bd.machine_threshold())
        self.assertAlmostEqual(b[0,1],     1/(5*sqrt(17)),    delta=10*bd.machine_threshold())
        self.assertAlmostEqual(b[1,0],     -9/(5*sqrt(17)),   delta=10*bd.machine_threshold())
        self.assertAlmostEqual(b[1,1],     -21/(10*sqrt(17)), delta=10*bd.machine_threshold())
        self.assertAlmostEqual(b[2,0],     -4/(5*sqrt(17)),   delta=10*bd.machine_threshold())
        self.assertAlmostEqual(b[2,1],     2/(5*sqrt(17)),    delta=10*bd.machine_threshold())

        nlft = StairlikeSequence2D(random_stairlike_sequence_2d(10, shape=(4, 4)), support_start=(2, 1))
        a, b = nlft.transform()
        self.assertAlmostEqual((a * a.conjugate() + b * b.conjugate() - 1).l2_norm(), 0, delta=10*bd.machine_threshold())

    def test_random_multiplication(self):
        nlft1 = StairlikeSequence2D(random_stairlike_sequence_2d(10, shape=(4, 4)))
        nlft2 = StairlikeSequence2D(random_stairlike_sequence_2d(10, shape=(4, 4)), support_start=(15, 10))

        a1, b1 = nlft1.transform()
        a2, b2 = nlft2.transform()

        r = b1 * b2

        z = (random_complex(1), random_complex(1))
        self.assertAlmostEqual(b1(z) * b2(z), r(z), delta=10 * bd.machine_threshold())

    def test_schwarz_transform(self):
        p = PolynomialMD(random_list(1, (3, 3)), support_start=(-1, -1))
        q = p.schwarz_transform()

        self.assertAlmostEqual(p[-1,-1]*2, q[-1,-1], delta=10*bd.machine_threshold())
        self.assertAlmostEqual(p[-1,0]*2,  q[-1,0],  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(0,          q[-1,1],  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(p[0,-1]*2,  q[0,-1],  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(p[0, 0],    q[0, 0],  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(0,          q[1,-1],  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(0,          q[0, 1],  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(0,          q[1, 0],  delta=10*bd.machine_threshold())
        self.assertAlmostEqual(0,          q[1, 1],  delta=10*bd.machine_threshold())

    def test_laurent_approx_md(self):
        N = 16
        points = random_list(1, (N, N)) # points[k][h] = f(\omega^k, \omega^h)

        P = weiss_md.laurent_approximation_md(points, 2)

        self.assertAlmostEqual(max(abs(P(bd.exp(2j*bd.pi()*k/N), bd.exp(2j*bd.pi()*h/N)) - points[k][h])
                                   for k in range(N) for h in range(N)), 0,
                                   delta=10*bd.machine_threshold())


if __name__ == '__main__':
    unittest.main()