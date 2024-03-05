import jax.numpy as jnp
from sngd.util.la import *

def cov2corr(V):
    v = jnp.sqrt(diagv(V))
    outer_v = outer(v, v)
    correlation = V / outer_v
    return correlation