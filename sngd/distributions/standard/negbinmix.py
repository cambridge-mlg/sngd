# a mixture of negative binomial distributions
import jax.numpy as jnp
from jax.random import split
from sngd.util.la import *
from tensorflow_probability.substrates.jax.distributions import NegativeBinomial

def mean(params):
    p, r, s = params
    return jnp.sum(p[:,None]*jax.scipy.stats.nbinom.mean(r, s), -2)

def var(params):
    p, r, s = params
    Ex = mean(params)
    Ex_i = jax.scipy.stats.nbinom.mean(r, s)
    v = jax.scipy.stats.nbinom.var(r, s)
    Exx = jnp.sum(p[...,:,None]*(v + Ex_i**2), -2)
    return Exx - Ex**2

def logprob(params, x):
    p = params[0]
    logp_i = vmap(lambda _: jax.scipy.stats.nbinom.logpmf(x, *_), out_axes=-1)(params[1:])
    return jax.scipy.special.logsumexp(jnp.log(p) + logp_i, -1)

def sample(key, params, nsamples):
    p, r, s = params
    k = p.shape[-1]
    key_w, key_x = split(key)
    x_i = vmap(lambda _: NegativeBinomial(_[0], probs=1-_[1]).sample((nsamples,), key_x))((r, s))
    mask = jax.nn.one_hot(jax.random.choice(key_w, k, (nsamples,), p=p), k, axis=0)
    return jnp.sum(mask*x_i, -2)