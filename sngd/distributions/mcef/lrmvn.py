# MCEF form of a low-rank (plus diagonal) multivariate normal distribution

import jax.numpy as jnp
from jax import vmap
from sngd.util.la import *

def stats(x):
    w, z = x
    return z, jnp.square(z), outer(z, w)

def standardparams(natparams):
    h, j, B = natparams
    v = -.5/j
    mu = v*h
    A = B*v[...,None]
    return mu, v, A

def natparams_from_standard(params):
    mu, v, A = params
    h = mu/v
    j = -.5/v
    B = A/v[...,None]
    return h, j, B

def meanparams(natparams):
    h, j, B = natparams
    dw = B.shape[-1]

    a = -2*j
    c = -B
    D = jnp.eye(dw) - .5*outer(B, B)/j[:,None,None]
    Dinv_c = syminvvp(D, c)

    # using block matrix inversion
    Ez = -.5*h/j
    Ezz = 1/(a - vdot(c, Dinv_c)) + jnp.square(Ez)
    EzwT = -Dinv_c/(a - vdot(c, Dinv_c))[:,None]

    return Ez, Ezz, EzwT

def natparams(meanparams):
    Ez, Ezz, EzwT = meanparams

    a = Ezz - jnp.square(Ez)
    c = EzwT

    # using sherman morrison
    B = vmap(lambda a_i, c_i:
        c_i/(a_i - vdot(c_i, c_i))
    )(a, c)
    j = vmap(lambda a_i, c_i:
        -.5/(a_i - vdot(c_i, c_i))
    )(a, c)
    h = -2 * Ez * j

    return h, j, B

def inmeandomain(meanparams):
    Ez, Ezz = meanparams[:2]
    return jnp.all(Ezz > jnp.square(Ez), -1)
