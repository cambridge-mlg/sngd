import jax
import jax.numpy as jnp
from sngd.distributions.ef import gamma
from sngd.util.tree import tree_sub

def logZ(natparams):
    return gamma.logZ(_to_gamma_natparams(natparams))

def dot(x1, x2):
    return x1[0]*x2[0] + x1[1]*x2[1]

def stats(x):
    return jnp.log(x), 1/x

def meanparams(natparams):
    return _from_gamma_meanparams(
        gamma.meanparams(
            _to_gamma_natparams(natparams)))

def kl(n1, n2):
    return dot(tree_sub(n1, n2), meanparams(n1)) + logZ(n2) - logZ(n1)

def logprob(natparams, x):
    a, b = standardparams(natparams)
    return a*jnp.log(b) - jax.scipy.special.gammaln(a) - (a+1)*jnp.log(x) - b/x

def mean(natparams):
    a, b = standardparams(natparams)
    return b/(a - 1)

def var(natparams):
    a, b = standardparams(natparams)
    return jnp.square(b)/(jnp.square(a-1)*(a-2))

def mode(natparams):
    a, b = standardparams(natparams)
    return b/(a + 1)

def standardparams(natparams):
    n1, n2 = natparams
    return -(n1+1), -n2

def sample(key, natparams, shape=()):
    a, b = standardparams(natparams)
    return 1/(jax.random.gamma(key, a, (*shape, *a.shape))/b)

def natparams(meanparams):
    return _from_gamma_natparams(
        gamma.natparams(
            _to_gamma_meanparams(meanparams)))

def natparams_from_standard(standardparams):
    a, b = standardparams
    return -(a+1), -b

def innaturaldomain(natparams):
    return gamma.innaturaldomain(_to_gamma_natparams(natparams))

def inmeandomain(meanparams):
    logx, x_inv = meanparams
    return (x_inv > 0) & (-logx < jnp.log(x_inv))

# convert to/from the params of 1/x, which is gamma distributed

def _to_gamma_natparams(natparams):
    return gamma.natparams_from_standard(
        standardparams(natparams))

def _from_gamma_natparams(natparams):
    return natparams_from_standard(
        gamma.standardparams(natparams))

def _to_gamma_meanparams(meanparams):
    logx, x_inv = meanparams
    return -logx, x_inv

def _from_gamma_meanparams(meanparams):
    logx, x = meanparams
    return -logx, x