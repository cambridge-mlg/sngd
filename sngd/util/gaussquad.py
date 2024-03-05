import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

def gaussquad(m, v, h, n):
    return jnp.sum(
        vmap(lambda x, w: w*h(jnp.sqrt(2*v)*x+m)/jnp.sqrt(jnp.pi))
            (*np.polynomial.hermite.hermgauss(n)),
        0)

def loggaussquadexp(m, v, h, n):
    return jax.scipy.special.logsumexp(
        vmap(lambda x, w: jnp.log(w) + h(jnp.sqrt(2*v)*x+m)/jnp.sqrt(jnp.pi))
            (*np.polynomial.hermite.hermgauss(n)),
        0)
