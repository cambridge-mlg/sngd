# the multivariate skew-t distribution of Azzalini (2013)
import jax
import jax.numpy as jnp
from jax.random import split
from sngd.distributions.ef import gamma
from sngd.distributions.standard import mvt, mvsn
from sngd.util.la import *
from sngd.util.stats import cov2corr
from tensorflow_probability.substrates.jax.distributions import StudentT

def mean(params):
    xi, Omega, eta, nu = params
    omega = jnp.sqrt(diagv(Omega))
    delta = mvsn._delta_from_eta(Omega, eta)
    b = jnp.sqrt(nu/jnp.pi)*jnp.exp(jax.scipy.special.gammaln(.5*(nu-1)) - jax.scipy.special.gammaln(.5*nu))
    return xi + b*omega*delta

def var(params):
    Omega, eta, nu = params[1:]
    omega = jnp.sqrt(diagv(Omega))
    delta = mvsn._delta_from_eta(Omega, eta)
    b = jnp.sqrt(nu/jnp.pi)*jnp.exp(jax.scipy.special.gammaln(.5*(nu-1)) - jax.scipy.special.gammaln(.5*nu))
    return nu/(nu-2)*Omega - (b**2)*outer(omega*delta, omega*delta)

def logprob(params, x):
    assert(x.ndim == 2)
    xi, Omega, eta, nu = params
    omega = jnp.sqrt(diagv(Omega))
    d = xi.shape[-1]
    R = diagm(1/omega) @ Omega @ diagm(1/omega)
    z = (x - xi)/omega
    Q = vdot(z, jnp.linalg.solve(R, z.T).T)
    alpha = mvsn._alpha_from_eta(Omega, eta)
    return (-jnp.sum(jnp.log(omega), -1) + jnp.log(2)
        + mvt.logprob((jnp.zeros(d), R, nu), z)
        + StudentT(nu + d, 0, 1).log_cdf(vdot(alpha, z)*jnp.sqrt((nu + d)/(nu + Q))))

def sample(key, params, nsamples):
    xi, Omega, eta, nu = params
    d = xi.shape[-1]

    key_w, key_z = split(key)
    w = gamma.sample(key_w, gamma.natparams_from_standard((.5*nu, .5)), (nsamples,))/nu

    alpha = mvsn._alpha_from_eta(Omega, eta)
    omega = jnp.sqrt(diagv(Omega))

    # alpha is invariant to rescaling
    Omega0 = cov2corr(Omega)
    eta0 = mvsn._eta_from_alpha(Omega0, alpha)
    xi0 = jnp.zeros(d)
    z0 = mvsn.sample(key_z, (xi0, Omega0, eta0), nsamples)
    z = xi + omega*z0/jnp.sqrt(w[:,None])

    return z