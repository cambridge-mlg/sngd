import jax
import jax.numpy as jnp
from jax import vmap
from jaxopt import FixedPointIteration
from sngd.util.controlflow import bounded_while_loop
from sngd.util.tree import tree_map, tree_scale, tree_sub

def logZ(natparams):
    n1, n2 = natparams
    return jax.scipy.special.gammaln(n1+1) - (n1+1)*jnp.log(-n2)

def dot(x1, x2):
    return x1[0]*x2[0] + x1[1]*x2[1]

def stats(x):
    return jnp.log(x), x

def meanparams(natparams):
    n1, n2 = natparams
    return jax.scipy.special.digamma(n1+1) - jnp.log(-n2), -(n1+1)/n2

def kl(n1, n2):
    return dot(tree_sub(n1, n2), meanparams(n1)) + logZ(n2) - logZ(n1)

def logprob(natparams, x):
    n1, n2 = natparams
    a, b = n1+1, -n2
    return a*jnp.log(b) - jax.scipy.special.gammaln(a) + (a-1)*jnp.log(x) - b*x

def mean(natparams):
    n1, n2 = natparams
    return -(n1+1)/n2

def var(natparams):
    n1, n2 = natparams
    return (n1+1)/(n2**2)

def mode(natparams):
    n1, n2 = natparams
    return -n1/n2

def standardparams(natparams):
    n1, n2 = natparams
    return n1+1, -n2

def sample(key, natparams, shape=()):
    a, b = standardparams(natparams)
    return jax.random.gamma(key, a, (*shape, *a.shape))/b

def natparams(meanparams):
    shape_prefix = meanparams[0].shape
    meanparams = tree_map(lambda _: _.reshape(-1, *_.shape[len(shape_prefix):]), meanparams)

    def _natparams(meanparams):
        # f is concave, strictly increasing, tends to -inf as b -> 0 from above, and has
        # exactly one root when meanparams lie in the domain of valid mean parameters.
        f = lambda m, a: jax.scipy.special.digamma(a) - jnp.log(a) + jnp.log(m[1]) - m[0]
        fprime = lambda a: jax.scipy.special.polygamma(1, a) - 1/a
        # some parameters inside the mean domain can fail to converge for numerical reasons
        alpha0 = bounded_while_loop(
            cond=lambda _: f(meanparams, _) >= 0,
            body=lambda _: .5*(_),
            init=2.0,
            maxiter=50)
        # use fixed point solver for efficient reverse-mode derivatives, but use our own
        # loop to find the starting point so that we can use a custom termination condition
        alpha = bounded_while_loop(
            cond=lambda _: jnp.abs(f(meanparams, _)) > 1e-12,
            body=lambda _: _ - f(meanparams, _)/fprime(_),
            init=alpha0,
            maxiter=50)
        solver = FixedPointIteration(lambda _, mp: _ - f(mp, _)/fprime(_), maxiter=1)
        alpha = solver.run(jax.lax.stop_gradient(alpha), mp=meanparams).params
        beta = alpha/meanparams[1]
        return natparams_from_standard((alpha, beta))
    
    _null_natparams = lambda _: tree_scale(_, jnp.nan)
    natparams = vmap(lambda _: jax.lax.cond(inmeandomain(_), _natparams, _null_natparams, _))(meanparams)
    return tree_map(lambda _: _.reshape(*shape_prefix, *_.shape[1:]), natparams)

def natparams_from_meanvar(m, v):
    b = m/v
    a = m*b
    return a-1, -b

def natparams_from_standard(standardparams):
    alpha, beta = standardparams
    return alpha-1, -beta

def innaturaldomain(natparams):
    n1, n2 = natparams
    return (n1 > -1) & (n2 < 0)

def inmeandomain(meanparams):
    logx, x = meanparams
    return (x > 0) & (logx < jnp.log(x))