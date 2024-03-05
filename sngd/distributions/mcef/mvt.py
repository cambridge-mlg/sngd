# MCEF form of a multivariate t distribution
import jax
import jax.numpy as jnp
from sngd.distributions.ef import invgamma, mvn
from sngd.util.la import outer

def stats(x):
    w, z = x
    z/w, outer(z, z)/w, -1/w - jnp.log(w)

def standardparams(natparams):
    h, J, a = natparams
    nu = 2*a
    m, V = mvn.standardparams((h, J))
    return m, V, nu

def standardparams_from_mean(meanparams):
    m1, m2, m3 = meanparams
    m, V = mvn.standardparams_from_mean((m1, m2))
    a = invgamma.standardparams(
        invgamma.natparams((-(m3+1), jnp.ones_like(m3))))[0]
    nu = 2*a
    return m, V, nu

def natparams_from_standard(params):
    mu, V, nu = params
    a = .5*nu
    h, J = mvn.natparams_from_standard((mu, V))
    return h, J, a

def meanparams(natparams):
    h, J, a = natparams
    m1, m2 = mvn.meanparams((h, J))
    m3 = -1 - jnp.log(a) + jax.scipy.special.digamma(a)
    return m1, m2, m3

def natparams(meanparams):
    m1, m2, m3 = meanparams
    h, J = mvn.natparams((m1, m2))
    a = invgamma.standardparams(
        invgamma.natparams((-(m3+1), jnp.ones_like(m3))))[0]
    return h, J, a

def innaturaldomain(natparams):
    h, J, a = natparams
    return mvn.innaturaldomain((h, J)) & (a > 0)

def inmeandomain(meanparams):
    m1, m2, m3 = meanparams
    return mvn.inmeandomain((m1, m2)) & invgamma.inmeandomain((-(m3+1), jnp.ones_like(m3)))

# convenience functions for converting to/from a minimal representation

def to_minimal_natparams(natparams):
    return (*mvn.to_minimal_natparams(natparams[:2]), natparams[2])

def from_minimal_natparams(natparams):
    return (*mvn.from_minimal_natparams(natparams[:2]), natparams[2])

def to_minimal_meanparams(meanparams):
    return (*mvn.to_minimal_meanparams(meanparams[:2]), meanparams[2])

def from_minimal_meanparams(meanparams):
    return (*mvn.from_minimal_meanparams(meanparams[:2]), meanparams[2])

def from_minimal_natparam_grad(g):
    return (*mvn.from_minimal_natparam_grad(g[:2]), g[2])

def from_minimal_meanparam_grad(g):
    return (*mvn.from_minimal_meanparam_grad(g[:2]), g[2])
