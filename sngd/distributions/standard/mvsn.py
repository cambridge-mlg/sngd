# the multivariate skew-normal distribution of Azzalini (2013)
import jax
import jax.numpy as jnp
from jax.random import split
from sngd.distributions.ef import mvn
from sngd.util.la import *
from sngd.util.gaussquad import gaussquad
from sngd.util.stats import cov2corr

def _delta_from_alpha(Omega, alpha):
    R = cov2corr(Omega)
    return mvp(R, alpha) / jnp.sqrt(1 + vdot(alpha, mvp(R, alpha)))[..., None]

def _delta_from_eta(Omega, eta):
    return _delta_from_alpha(Omega, _alpha_from_eta(Omega, eta))

def _alpha_from_delta(Omega, delta):
    R = cov2corr(Omega)
    Rinv_alpha = syminvvp(R, delta)
    return Rinv_alpha / jnp.sqrt(1 - vdot(delta, Rinv_alpha))

def _eta_from_delta(Omega, delta):
    return _eta_from_alpha(Omega, _alpha_from_delta(Omega, delta))

def _eta_from_alpha(Omega, alpha):
    omega = jnp.sqrt(diagv(Omega))
    return alpha/omega

def _alpha_from_eta(Omega, eta):
    omega = jnp.sqrt(diagv(Omega))
    return eta*omega

def _beta_from_eta(Omega, eta):
    omega = jnp.sqrt(diagv(Omega))
    return omega*_delta_from_eta(Omega, eta)

def mean(params):
    xi, Omega, eta = params
    omega = jnp.sqrt(diagv(Omega))
    b = jnp.sqrt(2/jnp.pi)
    return xi + b*omega*_delta_from_eta(Omega, eta)

def var(params):
    Omega, eta = params[1:]
    omega = jnp.sqrt(diagv(Omega))
    b = jnp.sqrt(2/jnp.pi)
    delta = _delta_from_eta(Omega, eta)
    return Omega - (b**2)*outer(omega*delta, omega*delta)

def logprob(params, x):
    xi, Omega, eta = params
    return (jnp.log(2)
        + jax.scipy.stats.norm.logcdf(vdot(eta, x-xi))
        + jax.scipy.stats.multivariate_normal.logpdf(x, xi, Omega))

def entropy(params):
    xi, Omega, eta = params
    norm = jax.scipy.stats.norm
    a = jnp.sqrt(vdot(eta, mvp(Omega, eta)))
    return (mvn.entropy_meanparams(mvn.meanparams_from_standard((xi, Omega)))
        -2*gaussquad(.0, 1., lambda _: norm.cdf(a*_)*(jnp.log(2) + norm.logcdf(a*_)), 100))

def sample(key, params, nsamples):
    key_w, key_x = split(key)
    w = jax.random.normal(key_w, (nsamples,))
    xi, Omega, eta = params
    beta = _beta_from_eta(Omega, eta)
    V = Omega - outer(beta, beta)
    return jax.random.multivariate_normal(key_x, xi + jnp.abs(w)[:,None]*beta, V, (nsamples,))