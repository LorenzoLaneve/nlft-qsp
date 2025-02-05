
import unittest

from tests import random_sequence

import mpmath as mp

from nlft_qsp.numeric.backend_mpmath import MPMathBackend
from nlft_qsp.numeric import bd, set_backend


class MPMathBackendTestCase(unittest.TestCase):

    def setUp(self):
        set_backend(MPMathBackend(mp.mp))
    
    @bd.workdps(10)
    def test_fft(self):
        seq = random_sequence(1, 256)
        iseq = bd.ifft(bd.fft(seq))

        self.assertAlmostEqual(max([bd.abs(x - y) for x, y in zip(iseq, seq)]), 0, delta=10 * bd.machine_eps())


        seq = random_sequence(10000, 256) # since |x| <= 10^4, the result of the fft gets degraded by 4 dps.
        iseq = bd.ifft(bd.fft(seq))

        self.assertAlmostEqual(max([abs(x - y) for x, y in zip(iseq, seq)]), 0, delta=10000 * bd.machine_eps())


if __name__ == '__main__':
    unittest.main()