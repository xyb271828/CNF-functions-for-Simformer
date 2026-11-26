import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import math
import numpy as np
from simformer_cold_atom import (jax,jnp,INIT,IDS,MASK,SCORE,velocity_and_divergence,
    integrate_logdensity,rk4_integrate,sigma,SIGMA_MAX,T_MIN,LOG_RATIO)


class SimformerTests(unittest.TestCase):
    def test_observed_generated_mask(self):
        np.testing.assert_array_equal(np.asarray(MASK)[0],np.r_[np.ones(9),np.zeros(5)])
    def test_official_model_has_fourteen_nodes_and_five_target_scores(self):
        params=INIT(jax.random.PRNGKey(0),jnp.ones(2),jnp.zeros((2,14,1)),IDS,MASK)
        scores=SCORE(params,jnp.ones(2)*.5,jnp.ones((2,5)),jnp.ones((2,9)))
        self.assertEqual(scores.shape,(2,5));self.assertTrue(np.isfinite(np.asarray(scores)).all())
    def test_target_only_divergence_matches_independent_full_jacobian(self):
        from simformer_cold_atom import velocity
        params=INIT(jax.random.PRNGKey(5),jnp.ones(2),jnp.zeros((2,14,1)),IDS,MASK)
        y=jnp.ones((2,5))*.4;c=jnp.ones((2,9))*.2
        _,div=velocity_and_divergence(params,.6,y,c)
        def single(x,context):return velocity(params,.6,x[None,:],context[None,:])[0]
        jac=jax.vmap(jax.jacfwd(single,argnums=0))(y,c)
        np.testing.assert_allclose(np.asarray(div),np.asarray(jnp.trace(jac,axis1=-2,axis2=-1)),rtol=2e-5,atol=2e-5)
    def test_likelihood_direction_and_five_dimensional_volume_term(self):
        y=jnp.ones((3,5))*.3;c=jnp.zeros((3,9));a=.2
        field=lambda t,x,c:(a*x,jnp.full((len(x),),5*a))
        z,integral=integrate_logdensity(field,y,c,0.,1.,32)
        nll=.5*jnp.sum(z*z,-1)+2.5*math.log(2*math.pi)-integral
        expected=.5*5*(.3*math.exp(a))**2+2.5*math.log(2*math.pi)-5*a
        np.testing.assert_allclose(np.asarray(nll),expected,atol=2e-6)
    def test_ve_probability_flow_with_analytic_gaussian_score(self):
        y=jnp.ones((2,5))*.7;c=jnp.zeros((2,9))
        def exact(t,x,c):
            rate=LOG_RATIO*sigma(t)**2/(1+sigma(t)**2)
            return rate*x,jnp.full((len(x),),5*rate)
        z,integral=integrate_logdensity(exact,y,c,T_MIN,1.,256)
        endstd=math.sqrt(1+SIGMA_MAX**2)
        nll=.5*jnp.sum((z/endstd)**2,-1)+5*math.log(endstd)+2.5*math.log(2*math.pi)-integral
        startvar=1+float(sigma(T_MIN))**2
        expected=.5*5*.7**2/startvar+2.5*math.log(2*math.pi*startvar)
        np.testing.assert_allclose(np.asarray(nll),expected,atol=1e-4)


if __name__=='__main__':unittest.main()
