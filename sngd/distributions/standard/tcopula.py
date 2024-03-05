import jax
import jax.numpy as jnp
import scipy.stats
from jax.random import split
from tensorflow_probability.substrates.jax.distributions import StudentT

from sngd.distributions.standard import mvt

def logprob(params, u):
    R, nu = params
    t = StudentT(nu, .0, 1.0)
    d = R.shape[-1]
    x = jnp.array(t.quantile(u))
    return mvt.logprob((jnp.zeros(d), R, nu), x) - jnp.array(t.log_prob(x)).sum(-1)

def sample(rng, params, n):
    R, nu = params
    d = R.shape[-1]
    rng_tau, rng_x = split(rng)
    alpha = beta = .5*nu
    tau = jax.random.gamma(rng_tau, alpha, (n,))/beta
    x = jax.random.multivariate_normal(rng_x, jnp.zeros(d), R/tau[:,None,None], (n,))
    return jnp.array(scipy.stats.t.cdf(x, nu))
