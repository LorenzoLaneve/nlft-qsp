import unittest

import numpy as np

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from nlft_qsp.poly import ChebyshevTExpansion, Polynomial
from nlft_qsp.rand import random_real_sequence, random_sequence
from nlft_qsp.qsp import ChebyshevQSPPhaseFactors, GQSPPhaseFactors, QSVTPhaseFactors, XQSPPhaseFactors, YQSPPhaseFactors


class SerializationTestCase(unittest.TestCase):

    def test_poly_serialize(self):
        with TemporaryDirectory() as temp_dir:
            P = Polynomial(random_sequence(10, 10), support_start=-2)

            P.dump_json(Path(temp_dir) / "poly.qspx")

            json_data = json.loads((Path(temp_dir) / "poly.qspx").read_text(encoding="utf-8"))

            Q = Polynomial.load_json(Path(temp_dir) / "poly.qspx")

            self.assertEqual(json_data.get("@type"), "@qspx/polynomial")
            self.assertTrue(P.support_start == Q.support_start)
            self.assertTrue(np.all(P.coeffs == Q.coeffs))

    def test_chebyshev_t_serialize(self):
        with TemporaryDirectory() as temp_dir:
            T = ChebyshevTExpansion(random_sequence(10, 10))
            T.dump_json(Path(temp_dir) / "cheb.qspx")

            json_data = json.loads((Path(temp_dir) / "cheb.qspx").read_text(encoding="utf-8"))
            S = ChebyshevTExpansion.load_json(Path(temp_dir) / "cheb.qspx")

            self.assertEqual(json_data.get("@type"), "@qspx/chebyshev_t")
            self.assertTrue(T.support_start == S.support_start)
            self.assertTrue(np.all(T.coeffs == S.coeffs))  

    def test_gqsp_serialize(self):
        with TemporaryDirectory() as temp_dir:
            T = GQSPPhaseFactors(phi=random_real_sequence(10, 10), lbd=0.4, theta=random_real_sequence(10, 10))
            T.dump_json(Path(temp_dir) / "phase_factors.qspx")

            json_data = json.loads((Path(temp_dir) / "phase_factors.qspx").read_text(encoding="utf-8"))
            S = GQSPPhaseFactors.load_json(Path(temp_dir) / "phase_factors.qspx")

            self.assertEqual(json_data.get("@type"), "@qspx/phase_factors/gqsp")
            self.assertTrue(all(T.phi[k] == S.phi[k] for k in range(len(T.phi))))   

    def test_qsp_serialize(self):
        with TemporaryDirectory() as temp_dir:
            for cls, type_tag in [
                (XQSPPhaseFactors, "@qspx/phase_factors/xqsp"),
                (YQSPPhaseFactors, "@qspx/phase_factors/yqsp"),
                (QSVTPhaseFactors, "@qspx/phase_factors/qsvt"),
                (ChebyshevQSPPhaseFactors, "@qspx/phase_factors/chebqsp")
            ]:
                T = cls(random_real_sequence(10, 10))
                T.dump_json(Path(temp_dir) / "phase_factors.qspx")

                json_data = json.loads((Path(temp_dir) / "phase_factors.qspx").read_text(encoding="utf-8"))
                S = cls.load_json(Path(temp_dir) / "phase_factors.qspx")

                self.assertEqual(json_data.get("@type"), type_tag)
                self.assertTrue(all(T.phi[k] == S.phi[k] for k in range(len(T.phi))))   


if __name__ == '__main__':
    unittest.main()