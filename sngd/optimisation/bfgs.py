import jax.numpy as jnp
from sngd.util.la import *
from sngd.util.tree import *

def init(params):
    step = 0
    H = jnp.eye(tree_vec(params).shape[-1])
    g = tree_scale(params, 0)
    p = tree_scale(params, 0)
    alpha = jnp.zeros(())
    return step, H, g, p, alpha

def update_pre(state, g):
    step, alpha = state[0], state[4]
    H = jax.lax.cond(step > 0, _update_H, lambda *_: _[0][1], state, g)    
    gflat, unvec = tree_vec(g, True)
    p = unvec(-H@gflat)
    state = step + 1, H, g, p, alpha
    return state

def search_direction(state):
    return state[3]

# alpha must satisfy the Wolfe conditions for the current search direction
def update_post(state, alpha):
    state = *state[:-1], alpha
    return state

def _update_H(state, gnext):
    H, g, p, alpha = state[1:]
    s = tree_vec(tree_scale(p, alpha))
    y = tree_vec(gnext) - tree_vec(g)
    sTy = s.dot(y)
    HysT = H @ outer(y, s)
    U = (sTy + y.dot(H @ y)) * outer(s, s) / jnp.square(sTy)
    V = (HysT + HysT.T)/sTy
    H = H + U - V
    return H