# MCEF form of a multivariate skew-normal distribution
import jax.numpy as jnp
from sngd.distributions.ef import mvn
from sngd.distributions.standard.mvsn import _delta_from_eta, _eta_from_delta
from sngd.util.la import diagv, invcholp, isposdefh, outer, mvp

# externally we expose standard parameters as (xi, Omega, eta) following Azzalini (2013).
# internally we use the notation of Lin (2019), (mu, V, beta, nu). we have changed
# alpha => beta here due to the clash of notation.
def _to_lin(params):
    mu, Omega, eta = params
    omega = jnp.sqrt(diagv(Omega))
    delta = _delta_from_eta(Omega, eta)
    beta = omega * delta
    V = Omega - outer(beta, beta)
    return mu, V, beta

def _from_lin(params):
    mu, V, beta = params
    Omega = V + outer(beta, beta)
    omega = jnp.sqrt(diagv(Omega))
    delta = beta / omega
    eta = _eta_from_delta(Omega, delta)
    return mu, Omega, eta

def stats(x):
    w, z = x
    z, outer(z, z), jnp.abs(w)*z

def standardparams(natparams):
    h, J, a = natparams
    m, V = mvn.standardparams((h, J))
    beta = mvp(V, a)
    return _from_lin((m, V, beta))

def natparams_from_standard(params):
    mu, V, beta = _to_lin(params)
    a = invcholp(jnp.linalg.cholesky(V), beta)
    h, J = mvn.natparams_from_standard((mu, V))
    return h, J, a

def meanparams(natparams):
    mu, V, beta = _to_lin(standardparams(natparams))
    c = jnp.sqrt(2/jnp.pi)
    m1 = mu + c*beta
    m2 = outer(mu, mu) + outer(beta, beta) + c*(outer(mu, beta) + outer(beta, mu)) + V
    m3 = c*mu + beta
    return m1, m2, m3

def natparams(meanparams):
    m1, m2, m3 = meanparams
    c = jnp.sqrt(2/jnp.pi)
    mu = (m1 - c*m3)/(1 - c**2)
    beta = m3 - c*mu
    V = m2 - outer(mu, mu) - outer(beta, beta) - c*(outer(mu, beta) + outer(beta, mu))
    return natparams_from_standard(_from_lin((mu, V, beta)))

def innaturaldomain(natparams):
    h, J = natparams[:2]
    return mvn.innaturaldomain((h, J))

def inmeandomain(meanparams):
    m1, m2, m3 = meanparams
    c = jnp.sqrt(2/jnp.pi)
    mu = (m1 - c*m3)/(1 - c**2)
    beta = m3 - c*mu
    V = m2 - outer(mu, mu) - outer(beta, beta) - c*(outer(mu, beta) + outer(beta, mu))
    return isposdefh(V)

# convenience functions for converting to/from a minimal representation

def to_minimal_natparams(natparams):
    return (*mvn.to_minimal_natparams(natparams[:2]), natparams[2])

def from_minimal_natparams(natparams):
    return (*mvn.from_minimal_natparams(natparams[:2]), natparams[2])

def to_minimal_meanparams(meanparams):
    return (*mvn.to_minimal_meanparams(meanparams[:2]), natparams[2])

def from_minimal_meanparams(meanparams):
    return (*mvn.from_minimal_meanparams(meanparams[:2]), natparams[2])

def from_minimal_natparam_grad(g):
    return (*mvn.from_minimal_natparam_grad(g[:2]), g[2])
