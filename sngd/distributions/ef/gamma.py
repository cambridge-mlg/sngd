import jax
import jax.numpy as jnp
from jax import vmap
from jaxopt import implicit_diff
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
    # handle batching over leading dimensions
    shape_prefix = meanparams[0].shape
    meanparams = tree_map(lambda _: _.reshape(-1, *_.shape[len(shape_prefix):]), meanparams)

    def _natparams(meanparams):
        # f is concave, strictly increasing, tends to -inf as b -> 0 from above, and has
        # exactly one root when meanparams lie in the domain of valid mean parameters.
        f = lambda a, mp: jax.scipy.special.digamma(a) - jnp.log(a) + jnp.log(mp[1]) - mp[0]
        fprime = lambda a: jax.scipy.special.polygamma(1, a) - 1/a
        # find a starting point alpha0 > 0 such that f is below 0. note that some parameters
        # inside the mean domain can fail to converge for numerical reasons, hence the bound.
        alpha0 = bounded_while_loop(
            cond=lambda _: f(_, meanparams) >= 0,
            body=lambda _: .5*(_),
            init=2.0,
            maxiter=100)
        # define custom_root condition to get efficient reverse-mode derivatives from jaxopt.
        @implicit_diff.custom_root(f)
        def _newton_solve(alpha0, mp):
            return bounded_while_loop(
                cond=lambda _: jnp.abs(f(_, mp)) > 1e-12,
                body=lambda _: _ - f(_, mp)/fprime(_),
                init=alpha0,
                maxiter=100)
        # newton's method is guaranteed to converge to a root of f from our starting point.
        alpha = _newton_solve(jax.lax.stop_gradient(alpha0), meanparams)
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
