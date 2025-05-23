
import unittest

import numpy as np

from nlft import NonLinearFourierSequence
import nlft_qsp.numerics as bd

from nlft_qsp.catalysts import catalyst_consumer, catalyst_polynomials, solve_adversary_bound
from nlft_qsp.rand import random_polynomial, random_sequence
from nlft_qsp import weiss


class CatalystConsumerTestCase(unittest.TestCase):

    def test_solve_adversary(self):
        n = 10

        P = random_polynomial(n+1, eta=0.5)
        Q = weiss.complete(P).shift(n)

        X = solve_adversary_bound(P, Q)
        for k in range(n+1):
            for h in range(n+1):

                if k == 0 and h == 0:
                    ekh = 1 - np.conj(P[k]) * P[h] - np.conj(Q[k]) * Q[h]
                else:
                    ekh = - np.conj(P[k]) * P[h] - np.conj(Q[k]) * Q[h]

                if k < n and h < n:
                    Xkh = X[k, h]
                else:
                    Xkh = 0

                if k > 0 and h > 0:
                    Xkhm = X[k-1, h-1]
                else:
                    Xkhm = 0

                self.assertAlmostEqual(ekh, Xkh - Xkhm, delta=bd.machine_threshold())
                # Checking \delta_{k,0} \delta_{h,0} - <\tau_k, \tau_h> = X[k, h] - X[k-1, h-1]
                # considering boundaries

    def test_catalyst_polynomials(self):
        n = 100

        P = random_polynomial(n+1, eta=0.5)
        Q = weiss.complete(P).shift(n)

        X = solve_adversary_bound(P, Q)

        for k in range(n):
            self.assertLessEqual(bd.abs(X[k, k]), 1)

        v = catalyst_polynomials(X)
        for k in range(n+1):
            for h in range(n+1):

                if k == 0 and h == 0:
                    ekh = 1 - np.conj(P[k]) * P[h] - np.conj(Q[k]) * Q[h]
                else:
                    ekh = - np.conj(P[k]) * P[h] - np.conj(Q[k]) * Q[h]

                if k < n and h < n:
                    Xkh = sum(np.conj(v[j][k]) * v[j][h] for j in range(n))
                else:
                    Xkh = 0

                if k > 0 and h > 0:
                    Xkhm = sum(np.conj(v[j][k-1]) * v[j][h-1] for j in range(n))
                else:
                    Xkhm = 0

                # print(bd.abs(ekh - (Xkh - Xkhm)))
                self.assertAlmostEqual(ekh, Xkh - Xkhm, delta=bd.machine_threshold())

    def test_catalyst_consumer(self):
        n = 5

        F = NonLinearFourierSequence(random_sequence(1, n))
        P, Q = F.transform()
        P = P.shift(n-1)

        F2 = catalyst_consumer(P, Q)

        for k in range(n):
            self.assertAlmostEqual(F[k], F2[k], delta=bd.machine_threshold(), msg=f"At {k}")

    def test_catalyst_consumer_outer(self):
        n = 150

        Q = random_polynomial(n+1, eta=0.5)
        P = weiss.complete(Q).shift(n)

        F = catalyst_consumer(P, Q)
        P2, Q2 = F.transform()

        self.assertAlmostEqual((Q - Q2).l2_norm(), 0, delta=bd.machine_threshold())

        




if __name__ == '__main__':
    unittest.main()