import jax.numpy as jnp
import numpy as np
import operator
from functools import partial
from jax import vmap
from jax.tree_util import tree_flatten, tree_map, tree_map, tree_unflatten

# tree utilities
tree_dropfirst = lambda _: tree_map(lambda x: x[1:], _)
tree_droplast = lambda _: tree_map(lambda x: x[:-1], _)
tree_first = lambda _: tree_map(lambda x: x[0], _)
tree_last = lambda _: tree_map(lambda x: x[-1], _)

def tree_append(xs, x, axis=0):
    x = tree_map(lambda _: jnp.expand_dims(_, axis), x)
    return tree_cat([xs, x], axis)

def tree_prepend(xs, x, axis=0):
    x = tree_map(lambda _: jnp.expand_dims(_, axis), x)
    return tree_cat([x, xs], axis)

def tree_cat(trees, axis=0):
    flats, treedefs = tuple(zip(*list(map(tree_flatten, trees))))
    flats = tuple(zip(*flats))
    tree = list(map(partial(jnp.concatenate, axis=axis), flats))
    return tree_unflatten(treedefs[0], tree)

def tree_stack(trees, axis=0):
    flats, treedefs = tuple(zip(*list(map(tree_flatten, trees))))
    flats = tuple(zip(*flats))
    tree = list(map(partial(jnp.stack, axis=axis), flats))
    return tree_unflatten(treedefs[0], tree)

def tree_sum(trees):
    flats, treedefs = tuple(zip(*list(map(tree_flatten, trees))))
    flats = tuple(zip(*flats))
    sums = list(map(sum, flats))
    return tree_unflatten(treedefs[0], sums)

def tree_mul(tree1, tree2):
    return tree_map(operator.mul, tree1, tree2)

def tree_add(tree1, tree2):
    return tree_map(operator.add, tree1, tree2)

def tree_sub(tree1, tree2):
    return tree_map(operator.sub, tree1, tree2)

def tree_mul(tree1, tree2):
    return tree_map(operator.mul, tree1, tree2)

def tree_mean(tree, axis=None):
    return tree_map(partial(jnp.mean, axis=axis), tree)

def tree_scale(tree, c):
    def _scale(_):
        _c = c if jnp.isscalar(c) else c.reshape(*c.shape + (1,)*(_.ndim-c.ndim))
        return _*_c
    return tree_map(_scale, tree)

def tree_mean(tree, axis):
    return tree_map(partial(jnp.mean, axis=axis), tree)

def tree_vec(tree, unvec=False):
    flat, treedef = tree_flatten(tree)
    shapes = list(map(jnp.shape, flat))
    lengths = list(map(lambda _: np.prod(_, dtype=np.int64), shapes))
    def _unvec(x):
        xs = np.split(x, np.cumsum(np.array(lengths[:-1])))
        flat = list(map(lambda _: _[0].reshape(_[1]), zip(xs, shapes)))
        return tree_unflatten(treedef, flat)
    x = jnp.concatenate(list(map(partial(jnp.reshape, newshape=-1), flat)))
    return (x, _unvec) if unvec else x

def tree_interpolate(a, b, x):
    return tree_add(tree_scale(a, 1-x), tree_scale(b, x))

def tree_where(cond, a, b):
    return tree_add(tree_scale(a, cond), tree_scale(b, 1-cond))

def tree_len(x, axis=0):
    return jnp.sum(vmap(lambda _: 1, axis)(x))