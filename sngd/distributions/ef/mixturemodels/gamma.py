import jax
import jax.numpy as jnp
from sngd.distributions.ef import gamma
from sngd.util.la import *

def meanparams(natparams):
    nu, n1, n2 = natparams
    logZ = gamma.logZ((n1, n2))
    p = jnp.exp(nu + logZ - jax.scipy.special.logsumexp(nu + logZ))
    m = gamma.meanparams((n1, n2))
    return p, p*m[0], p*m[1]

def natparams(meanparams):
    p, pm1, pm2 = meanparams
    m1, m2 = pm1/p, pm2/p
    n1, n2 = gamma.natparams((m1, m2))
    nu = jnp.log(p) - gamma.logZ((n1, n2))
    return nu, n1, n2

def standardparams(natparams):
    nu = natparams[0]
    logZ_i = gamma.logZ(natparams[1:])
    p = jnp.exp(_lognormalise(nu + logZ_i))
    alpha, beta = gamma.standardparams(natparams[1:])
    return p, alpha, beta

def natparams_from_standard(params):
    p = params[0]
    n1, n2 = gamma.natparams_from_standard(params[1:])
    nu = jnp.log(p) - gamma.logZ((n1, n2))
    return nu, n1, n2

def clip_weights(natparams, lb):
    nu = natparams[0]
    logZ_i = gamma.logZ(natparams[1:])
    p = jnp.exp(_lognormalise(nu + logZ_i))
    p = jnp.clip(p, lb)
    p /= jnp.sum(p, -1, keepdims=True)
    nu = jnp.log(p) - logZ_i
    return nu, *natparams[1:]

def clip_weights_meanparams(meanparams, lb):
    p0, pm1, pm2 = meanparams
    p = jnp.clip(p0, lb)
    p /= jnp.sum(p, -1, keepdims=True)
    return p, pm1*p/p0, pm2*p/p0

def logZ(natparams):
    nu = natparams[0]
    logZ = gamma.logZ(natparams[1:])
    assert(nu.shape == logZ.shape)
    return jax.scipy.special.logsumexp(nu + logZ)

def innaturaldomain(natparams):
    return jnp.all(gamma.innaturaldomain(natparams[1:]), -1)

def inmeandomain(meanparams):
    p, pm1, pm2 = meanparams
    m1, m2 = pm1/p, pm2/p
    return jnp.all(p >= .0, -1) & jnp.allclose(p.sum(-1), 1.0) & jnp.all(gamma.inmeandomain((m1, m2)), -1)

_lognormalise = lambda _: _ - jax.scipy.special.logsumexp(_, -1)

# convenience functions for converting to/from a minimal representation

def to_minimal_natparams(natparams):
    nu = natparams[0]
    nu = nu[...,:-1] - nu[...,-1]
    return nu, *natparams[1:]

def from_minimal_natparams(natparams):
    nu = natparams[0]
    logZ_i = gamma.logZ(natparams[1:])
    logp_ratio = nu + logZ_i[...,:-1] - logZ_i[...,-1]
    logp_k = -jax.scipy.special.logsumexp(logp_ratio, -1)
    logp_i = logp_ratio + logp_k
    nu = jnp.concatenate([logp_i, logp_k[...,None]], -1) - logZ_i
    return nu, *natparams[1:]

def to_minimal_meanparams(meanparams):
    p = meanparams[0]
    return p[...,:-1], *meanparams[1:]

def from_minimal_meanparams(meanparams):
    p = meanparams[0]
    p = jnp.concatenate([p, 1-jnp.sum(p, -1, keepdims=True)], -1)
    return p, *meanparams[1:]

def to_minimal_natparam_grad(g):
    g_nu = g[0]
    g_nu = g_nu[...,:-1] - g_nu[...,-1]
    return g_nu, *g[1:]

def from_minimal_meanparam_grad(g):
    g_p = g[0]
    g_p = jnp.concatenate([g_p, -jnp.sum(g_p, -1, keepdims=True)], -1)
    return g_p, *g[1:]