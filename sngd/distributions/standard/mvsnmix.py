# a mixture of multivariate skew-normal distributions
import jax.numpy as jnp
from jax.random import split
from sngd.distributions.standard import mvsn
from sngd.util.la import *

def mean(params):
    p, xi, Omega, eta = params
    Ex = mvsn.mean((xi, Omega, eta))
    return jnp.sum(p[:,None]*Ex, -2)

def var(params):
    p, xi, Omega, eta = params
    Ex = mean(params)
    Ex_i = mvsn.mean((xi, Omega, eta))
    Exx = jnp.sum(p[...,:,None,None]*(mvsn.var((xi, Omega, eta)) + outer(Ex_i, Ex_i)), -3)
    return Exx - outer(Ex, Ex)

def logprob(params, x):
    p = params[0]
    logp_i = vmap(mvsn.logprob, (0, None))(params[1:], x)
    return jax.scipy.special.logsumexp(jnp.log(p) + logp_i, -1)

def sample(key, params, nsamples):
    p = params[0]
    k = p.shape[-1]
    key_w, key_x = split(key)
    x_i = vmap(lambda _: mvsn.sample(key_x, _, nsamples))(params[1:])
    mask = jax.nn.one_hot(jax.random.choice(key_w, k, (nsamples,), p=p), k, axis=0)
    return jnp.sum(mask[...,:,None]*x_i, -3)
