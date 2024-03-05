import jax
import jax.numpy as jnp
import numpy as np
import math
from jax import vmap

# batched operations 
outer = lambda x, y: x[...,None]*y[...,None,:]
transpose = lambda _: jnp.swapaxes(_, -1, -2)
mvp = lambda X, v: jnp.matmul(X, v[...,None]).squeeze(-1)
mmp = lambda X, Y: jnp.matmul(X, Y)
vmp = lambda v, X: jnp.matmul(v[...,None,:], X).squeeze(-2)
vdot = lambda x, y: jnp.sum(x*y, -1)

def diagm(x):
    *shape_prefix, D = x.shape
    nbatches = np.prod(shape_prefix, dtype=np.int64)
    X = vmap(jnp.diag)(x.reshape(nbatches,D))
    return X.reshape(*shape_prefix, D, D)

def diagv(X):
    *shape_prefix, D = X.shape[:-1]
    nbatches = np.prod(shape_prefix, dtype=np.int64)
    V = vmap(jnp.diag)(X.reshape(nbatches,D,D))
    return V.reshape(*shape_prefix, D)

def trilm(x):
    shape_prefix = x.shape[:-1]
    nbatches = np.prod(shape_prefix, dtype=np.int64)
    D = (-1 + math.isqrt(1 + 8*x.shape[-1]))//2
    X = vmap(lambda _: jnp.zeros((D,D)).at[jnp.tril_indices(D)].set(_))(x.reshape(nbatches, -1))
    return X.reshape(*shape_prefix, D, D)

def trilv(X):
    *shape_prefix, D = X.shape[:-1]
    nbatches = np.prod(shape_prefix, dtype=np.int64)
    x = vmap(lambda _: _[jnp.tril_indices(D)])(X.reshape(nbatches, D, D))
    return x.reshape(*shape_prefix, -1)

# inv(L*L.T)*Y
def invcholp(L, Y):
    D = jax.scipy.linalg.solve_triangular(L, Y, lower=True)
    B = jax.scipy.linalg.solve_triangular(transpose(L), D, lower=False)
    return B

# inv(L*L.T)
def invchol(L):
    return invcholp(L, jnp.tile(jnp.eye(L.shape[-1]), (*L.shape[:-2],1,1)))

def syminvvp(H, y):
    Hchol = jnp.linalg.cholesky(H)
    return invcholp(Hchol, y[...,None]).squeeze(-1)

def isposdefh(X):
    return jnp.linalg.eigvalsh(X)[...,0] > .0
