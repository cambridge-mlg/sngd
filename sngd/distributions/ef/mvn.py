import jax
import jax.numpy as jnp
from sngd.util.la import *
from sngd.util.tree import tree_sub

def logZ(natparams):
    L = jnp.linalg.cholesky(-2*natparams[1])
    return logZcholP(natparams, L)

def logZcholP(natparams, L):
    h = natparams[0]
    v = jax.scipy.linalg.solve_triangular(L, h, lower=True)
    halflogdet = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-1, axis2=-2)), -1)
    return .5*h.shape[-1]*jnp.log(2*jnp.pi) + .5*vdot(v, v) - halflogdet

def logZcholV(natparams, L):
    h = natparams[0]
    v = mvp(transpose(L), h)
    halflogdet = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-1, axis2=-2)), -1)
    return .5*h.shape[-1]*jnp.log(2*jnp.pi) + .5*vdot(v, v) + halflogdet

def logp(natparams, x):
    return dot(natparams, stats(x)) - logZ(natparams)

def sample(key, natparams, shape=()):
    mu, V = standardparams(natparams)
    return jax.random.multivariate_normal(key, mu, V, shape)

def natparams(meanparams):
    x, xx = symmetrise(meanparams)
    J = -.5*jnp.linalg.inv(xx - outer(x,x))
    J = .5 * (J + transpose(J))
    h = -2*mvp(J, x)
    return h, J

def meanparams(natparams):
    h, J = symmetrise(natparams)
    J = .5*(J + transpose(J))
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    return mu, V + outer(mu, mu)

def standardparams(natparams):
    h, J = symmetrise(natparams)
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    return mu, V

def standardparams_from_mean(meanparams):
    x, xx = meanparams
    return x, xx - outer(x, x)

def mean(natparams):
    return standardparams(natparams)[0]

def var(natparams):
    return standardparams(natparams)[1]

def natparams_from_standard(params):
    m, V = symmetrise(params)
    J = -.5*jnp.linalg.inv(V)
    h = -2*mvp(J, m)
    return h, J

def meanparams_from_standard(standardparams):
    mu, V = standardparams
    return mu, V + outer(mu, mu)

def stats(x):
    return x, outer(x, x)

def dot(natparams, stats):
    h, J = natparams
    x, xx = stats
    return jnp.sum(h*x, axis=-1) + jnp.sum(J*xx, axis=(-1,-2))

def kl(natparams1, natparams2):
    return logZ(natparams2) - logZ(natparams1) \
        + dot(tree_sub(natparams1, natparams2), meanparams(natparams1))

def kl_meanparams(meanparams1, meanparams2):
    natparams1, natparams2 = natparams(meanparams1), natparams(meanparams2)
    return logZ(natparams2) - logZ(natparams1) \
        + dot(tree_sub(natparams1, natparams2), meanparams1)

def crossentropy(natparams1, natparams2):
    return logZ(natparams2) - dot(natparams2, meanparams(natparams1))

def innaturaldomain(natparams):
    return isposdefh(-2*natparams[1])

def inmeandomain(meanparams):
    return isposdefh(meanparams[1])

def symmetrise(natparams):
    h, J = natparams
    return h, .5*(J + transpose(J))

def entropy_meanparams(meanparams):
    x, xx = meanparams
    D = x.shape[-1]
    V = xx - outer(x, x)
    return .5 * D * (1 + jnp.log(2 * jnp.pi)) + .5 * jnp.linalg.slogdet(V)[1]

# convenience functions for converting to/from a minimal representation

def to_minimal_natparams(_):
    h, J = _
    J = trilv(J + J.T - diagm(diagv(J)))
    return h, J

def from_minimal_natparams(_):
    h, J = _
    J = trilm(J)
    J = .5*(J + J.T)
    return h, J

def to_minimal_meanparams(_):
    x, xx = _
    return x, trilv(xx)

def from_minimal_meanparams(_):
    x, xx = _
    xx = trilm(xx)
    xx = (xx + transpose(xx) - diagm(diagv(xx)))
    return x, xx

def from_minimal_natparam_grad(_):
    h, J = _
    J = trilm(J)
    J = .5*(J + J.T)
    return h, J

def from_minimal_meanparam_grad(_):
    x, xx = _
    xx = trilm(xx)
    xx = (xx + transpose(xx) - diagm(diagv(xx)))
    return x, xx
