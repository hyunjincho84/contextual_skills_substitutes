import importlib.util
from pathlib import Path
import unittest
import tempfile

import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location("dispersion",Path(__file__).with_name("python_yearly_dispersion.py"))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PairwiseCosineTest(unittest.TestCase):
    def test_exact_identity_matches_all_pairs_and_ignores_self_pairs(self):
        values=np.array([[2.,0.,0.],[0.,3.,0.],[4.,4.,0.],[1.,2.,3.]])
        unit=values/np.linalg.norm(values,axis=1,keepdims=True)
        expected=np.mean([unit[i]@unit[j] for i in range(4) for j in range(i+1,4)])
        observed=module.cosine_summary(values)
        self.assertAlmostEqual(observed["mean_pairwise_cosine"],expected)
        self.assertEqual(observed["n_unordered_pairs"],6)
        self.assertAlmostEqual(observed["mean_cosine_distance"],1-expected)

    def test_normalizes_posting_vectors_and_records_zero_norm(self):
        stats=module.cosine_summary(np.array([[2.,0.],[100.,0.],[0.,0.]]))
        self.assertEqual(stats["n_zero_norm_postings"],1)
        self.assertEqual(stats["n_valid_postings"],2)
        self.assertAlmostEqual(stats["mean_pairwise_cosine"],1.)

    def test_too_few_valid_postings_has_no_defined_pair_average(self):
        stats=module.cosine_summary(np.array([[2.,0.],[0.,0.]]))
        self.assertEqual(stats["n_unordered_pairs"],0)
        self.assertTrue(np.isnan(stats["mean_pairwise_cosine"]))

    def test_draws_overall_and_occupation_figures(self):
        rows=[]
        for scope in ["all_python","15-1252.00"]:
            for year in [2010,2011,2012]:
                rows.append({"scope":scope,"posted_year":year,"mean_cosine_distance":.2,
                             "n_valid_postings":10})
        selected=pd.DataFrame([{"onet_code":"15-1252.00","onet_name":"Software Developers"}])
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            module.draw_results(pd.DataFrame(rows),selected,output,40)
            self.assertTrue((output/"overall_dispersion.png").is_file())
            self.assertTrue((output/"onet_dispersion.png").is_file())


if __name__=="__main__": unittest.main()
