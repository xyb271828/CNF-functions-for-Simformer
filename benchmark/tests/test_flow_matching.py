import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from torch import nn
from flow_matching import VelocityMLP,exact_velocity_divergence,integrate_samples,nll_ode


class LinearVelocity(nn.Module):
    def __init__(self,a,b):super().__init__();self.a=a;self.b=b
    def forward(self,x,t,context):return self.a*x+self.b+context[:,:5]*0


class FlowMatchingTests(unittest.TestCase):
    def test_exact_divergence_for_linear_field(self):
        x=torch.randn(7,5,dtype=torch.float64);c=torch.zeros(7,9,dtype=x.dtype)
        _,d=exact_velocity_divergence(LinearVelocity(.3,1.),x,.5,c)
        torch.testing.assert_close(d,torch.full((7,),1.5,dtype=x.dtype))
    def test_translation_sampling_and_likelihood_sign(self):
        z=torch.randn(7,5,dtype=torch.float64);c=torch.zeros(7,9,dtype=z.dtype)
        m=LinearVelocity(0.,.7);x=integrate_samples(m,z,c,8)
        torch.testing.assert_close(x,z+.7)
        expected=.5*z.square().sum(-1)+2.5*math.log(2*math.pi)
        np.testing.assert_allclose(nll_ode(m,x,c,8),expected.numpy(),atol=1e-9)
    def test_scaling_likelihood_includes_volume_change(self):
        z=torch.randn(7,5,dtype=torch.float64);c=torch.zeros(7,9,dtype=z.dtype)
        a=.3;x=z*math.exp(a);expected=.5*z.square().sum(-1)+2.5*math.log(2*math.pi)+5*a
        np.testing.assert_allclose(nll_ode(LinearVelocity(a,0.),x,c,16),expected.numpy(),atol=1e-8)
    def test_time_context_and_state_shapes(self):
        m=VelocityMLP();x=torch.randn(7,5);c=torch.randn(7,9)
        self.assertEqual(tuple(m(x,.3,c).shape),(7,5))
        _,d=exact_velocity_divergence(m,x,torch.rand(7,1),c)
        self.assertTrue(torch.isfinite(d).all())


if __name__=='__main__':unittest.main()
