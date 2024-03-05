import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import while_loop
from sngd.util.tree import tree_add, tree_scale

def backtracking_linesearch(cond, x0, p, alphamax, tau=.5, maxiter=20):
    x = tree_add(x0, tree_scale(p, alphamax))
    alpha, niter = while_loop(
        lambda _: (_[2] <= maxiter) & ~cond(*_[:-1]),
        lambda _: (tree_add(x0, tree_scale(p, tau*_[1])), tau*_[1], _[2]+1),
        (x, alphamax, 0))[1:]
    return jax.lax.cond(
        niter <= maxiter,
        lambda: alpha,
        lambda: jnp.nan)

def grid_linesearch(x0, p, f, alphamax, n):
    stepsizes = jnp.linspace(0, alphamax, n+1)[1:]
    fvals = vmap(lambda _: f(tree_add(x0, tree_scale(p, _))))(stepsizes)
    minidx = jnp.nanargmin(fvals)
    return jax.lax.cond(
        minidx != -1,
        lambda: stepsizes[minidx],
        lambda: jnp.nan)
