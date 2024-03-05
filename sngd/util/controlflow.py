import jax
import jax.numpy as jnp
from sngd.util.tree import tree_scale

def bounded_while_loop(cond, body, init, maxiter):
    res = jax.lax.while_loop(
        lambda _: (_[1] < maxiter) & cond(_[0]),
        lambda _: (body(_[0]), _[1]+1),
        (init, 0))[0]
    return jax.lax.cond(cond(res), lambda _: tree_scale(_, jnp.nan), lambda _: _, res)