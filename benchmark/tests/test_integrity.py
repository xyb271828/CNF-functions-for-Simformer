"""Tests of data correctness and conditional sampling semantics, not metric targets."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import torch
from benchmark import CONTEXT, safe_join, balanced_weights, group_bootstrap_delta, make_flow

class IntegrityTests(unittest.TestCase):
    def test_ambiguous_keys_are_excluded_not_expanded(self):
        params=pd.DataFrame([dict(simulation=i,**dict.fromkeys(CONTEXT,v)) for i,v in [(1,0.),(1,1.),(2,2.)]])
        joined,ambiguous=safe_join(pd.DataFrame({'simulation':[1,2,2],'x':[10,20,30]}),params)
        self.assertEqual(ambiguous,{1});self.assertEqual(joined.x.tolist(),[20,30])
    def test_simulations_have_equal_total_weight(self):
        groups=np.array([1,1,1,2]);weights=balanced_weights(groups)
        self.assertAlmostEqual(weights[groups==1].sum(),weights[groups==2].sum())
    def test_paired_bootstrap_preserves_constant_difference(self):
        a=pd.Series([3.,8.,2.],index=[1,2,3]);r=group_bootstrap_delta(a,a-.5)
        np.testing.assert_allclose(r['paired_simulation_bootstrap_95_ci'],[.5,.5])
    def test_context_batch_produces_one_sample_per_context(self):
        torch.manual_seed(11);m=make_flow(True);c=torch.randn(7,9)
        with torch.no_grad():
            y,lp=m.sample(7,c);recomputed=m.log_prob(y,c)
        self.assertEqual(tuple(y.shape),(7,5))
        self.assertTrue(torch.isfinite(recomputed).all())
        torch.testing.assert_close(lp,recomputed,atol=3e-4,rtol=3e-4)

if __name__=='__main__':unittest.main()
