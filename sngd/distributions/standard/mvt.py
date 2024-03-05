# the multivariate t distribution
import jax
import jax.numpy as jnp
from jax.random import split
from sngd.distributions.ef import invgamma
from sngd.util.la import *

def mean(params):
    return params[0]

def var(params):
    V, nu = params[1:]
    return V*nu/(nu-2)

def logprob(params, x):
    mu, V, nu = params
    d = mu.shape[-1]
    Vchol = jnp.linalg.cholesky(V)
    return jax.scipy.special.gammaln(.5*(nu + d)) - \
        jax.scipy.special.gammaln(.5*nu) - .5*d*jnp.log(nu*jnp.pi) - \
        jnp.sum(jnp.log(diagv(Vchol)), -1) - \
        .5*(nu + d)*jnp.log(1 + vdot(x - mu, invcholp(Vchol, (x - mu).T).T)/nu)

def entropy(params):
    mu, V, nu = params
    D = mu.shape[-1]
    return .5*jnp.linalg.slogdet(V)[1] - jax.scipy.special.gammaln(.5*(nu + D)) \
        + jax.scipy.special.gammaln(.5*nu) + .5*D*jnp.log(nu*jnp.pi) \
        + .5*(nu + D)*(jax.scipy.special.digamma(.5*(nu + D)) - jax.scipy.special.digamma(.5*nu))

def sample(key, params, nsamples):
    mu, V, nu = params
    key_tau, key_X = split(key)
    w_natparams = invgamma.natparams_from_standard((.5*nu, .5*nu))
    w = invgamma.sample(key_tau, w_natparams, (nsamples,))
    return jax.random.multivariate_normal(key_X, mu, V*w[:,None,None], (nsamples,))
