import jax
import jax.numpy as jnp
from sngd.distributions.ef import mvn
from sngd.util.la import *

def meanparams(natparams):
    nu, h, J = natparams
    J = .5*(J + transpose(J))
    logZ = mvn.logZ((h, J))
    p = jnp.exp(nu + logZ - jax.scipy.special.logsumexp(nu + logZ))
    m = mvn.meanparams((h, J))
    return p, p[...,None]*m[0], p[...,None,None]*m[1]

def natparams(meanparams):
    p, px, pxx = meanparams
    x, xx = px/p[...,None], pxx/p[...,None,None]
    h, J = mvn.natparams((x, xx))
    nu = jnp.log(p) - mvn.logZ((h, J))
    return nu, h, J

def standardparams(natparams):
    nu, h, J = natparams
    logZ_i = mvn.logZ((h, J))
    p = jnp.exp(_lognormalise(nu + logZ_i))
    mu, V = mvn.standardparams((h, J))
    return p, mu, V

def standardparams_from_mean(meanparams):
    p, px, pxx = meanparams
    x, xx = px/p[...,None], pxx/p[...,None,None]
    mu = x
    V = xx - outer(x, x)
    return p, mu, V

def natparams_from_standard(params):
    p, mu, V = params
    h, J = mvn.natparams_from_standard((mu, V))
    nu = jnp.log(p) - mvn.logZ((h, J))
    return nu, h, J

def clip_weights(natparams, lb):
    nu, h, J = natparams
    logZ_i = mvn.logZ((h, J))
    p = jnp.exp(_lognormalise(nu + logZ_i))
    p = jnp.clip(p, lb)
    p /= jnp.sum(p, -1, keepdims=True)
    nu = jnp.log(p) - logZ_i
    return nu, h, J

def clip_weights_meanparams(meanparams, lb):
    p, px, pxx = meanparams
    x, xx = px/p[...,None], pxx/p[...,None,None]
    p = jnp.clip(p, lb)
    p /= jnp.sum(p, -1, keepdims=True)
    px = x*p[...,None]
    pxx = xx*p[...,None,None]
    return p, px, pxx

def logZ(natparams):
    nu, h, J = natparams
    logZ = mvn.logZ((h, J))
    assert(nu.shape == logZ.shape)
    return jax.scipy.special.logsumexp(nu + logZ)

def innaturaldomain(_):
    return jnp.all(mvn.innaturaldomain(_[1:]), -1)

def inmeandomain(_):
    p, px, pxx = _
    x, xx = px/p[...,None], pxx/p[...,None,None]
    return jnp.all(p >= .0, -1) & jnp.allclose(jnp.sum(p, -1), 1.0) & jnp.all(mvn.inmeandomain((x, xx)), -1)

def symmetrise(_):
    return _[0], *vmap(mvn.symmetrise)(_[1:])

_lognormalise = lambda _: _ - jax.scipy.special.logsumexp(_, -1)