import jax
import jax.numpy as jnp

logaddexp = jnp.logaddexp

def logsubexp(x, y):
    c = jnp.minimum(x, y)
    return jnp.log(jnp.exp(x-c) - jnp.exp(y-c)) + c

def logmeanexp(x, axis):
    return jax.scipy.special.logsumexp(x, axis=axis) - jnp.log(x.shape[axis])