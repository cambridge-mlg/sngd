import argparse
import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers
from sngd.distributions.ef.mixturemodels import gamma as gamma_mixturemodel
from sngd.distributions.standard import negbinmix
from sngd.optimisation import bfgs, linesearch
from sngd.util.la import *
from sngd.util.random import rngcall
from sngd.util.tree import tree_add, tree_scale
from jax.random import split

jax.config.update("jax_enable_x64", True)

def _generate_task(seed, k, n):
    rng = jax.random.PRNGKey(seed)
    key_params, key_x = split(rng)
    params = _random_params(key_params, k, mmin=5, mmax=5_000, overdispersion=3)
    x = negbinmix.sample(key_x, params, n)
    return x

def _random_params(key, k, mmin, mmax, overdispersion):
    key_p, key_m = split(key)
    p = jax.nn.softmax(jax.random.normal(key_p, (k,))*.5)
    m = jax.random.uniform(key_m, (k,), minval=mmin, maxval=mmax)
    v = m*jnp.square(overdispersion)
    s = m/v
    r = m*s/(1-s)
    return p, r, s

def _init_params(key, k, mmin, mmax):
    key_m, key_v = split(key)
    p = jnp.ones(k)/k
    m = jax.random.uniform(key_m, (k,), minval=mmin, maxval=mmax)
    v = m/jax.random.uniform(key_v, (k,), minval=.001, maxval=.02)
    s = m/v
    r = m*s/(1-s)
    return p, r, s

def _gamma_to_negbin(_):
    a, b = _
    in_domain = (b > .0) & (b < 1.) & (a > .0)
    s = jnp.where(in_domain, b, jnp.nan)
    r = jnp.where(in_domain, a/(1-s), jnp.nan)
    return r, s

def _negbin_to_gamma(_):
    r, s = _
    b = s
    a = r*(1-b)
    return a, b

def _sngd_init(args, init_params):

    meanparams = gamma_mixturemodel.meanparams(
        gamma_mixturemodel.natparams_from_standard(
            (init_params[0], *_negbin_to_gamma(init_params[1:]))))
    
    state = meanparams, 0

    def step(state, f):
        return _sngd_step(args, state, f)

    def params(state):
        meanparams = state[0]
        p, alpha, beta = gamma_mixturemodel.standardparams(gamma_mixturemodel.natparams(meanparams))
        return p, *_gamma_to_negbin((alpha, beta))

    return state, step, params

def _sngd_step(args, state, f):
    meanparams, opt_step = state

    def _f_minimal_natparams(natparams):
        p, alpha, beta = gamma_mixturemodel.standardparams(
            gamma_mixturemodel.from_minimal_natparams(natparams))
        return f((p, *_gamma_to_negbin((alpha, beta))))
    
    loss, g = jax.value_and_grad(_f_minimal_natparams)(
        (gamma_mixturemodel.to_minimal_natparams(
            gamma_mixturemodel.natparams(meanparams))))

    g = gamma_mixturemodel.from_minimal_meanparam_grad(g)

    def _f_meanparams(meanparams):
        natparams = gamma_mixturemodel.natparams(meanparams)
        p, alpha, beta = gamma_mixturemodel.standardparams(natparams)
        return f((p, *_gamma_to_negbin((alpha, beta))))
    
    p = tree_scale(g, -1)
    alpha = linesearch.grid_linesearch(meanparams, p, _f_meanparams, 1.0, args.nlinesearch)
    meanparams = tree_add(meanparams, tree_scale(p, alpha))

    state = meanparams, opt_step + 1

    return state, loss

def _bfgs_init(args, init_params):
    p, r, s = init_params

    params = jnp.log(p), jnp.log(r), jax.scipy.special.logit(s)

    state = bfgs.init(params), params

    def _step(state, f):
        return _bfgs_step(args, state, f)

    def _params(state):
        params = state[1]
        logits, logr, logs = params
        p, r, s = jax.nn.softmax(logits), jnp.exp(logr), jax.scipy.special.expit(logs)
        return p, r, s

    return state, _step, _params

def _bfgs_step(args, state, f):

    opt_state, params = state

    def _f(_):
        logits, logr, logs = _
        p, r, s = jax.nn.softmax(logits), jnp.exp(logr), jax.scipy.special.expit(logs)
        return f((p, r, s))

    loss, g = jax.value_and_grad(_f)(params)
    opt_state = bfgs.update_pre(opt_state, g)
    p = bfgs.search_direction(opt_state)
    
    alpha = linesearch.grid_linesearch(params, p, _f, 1.0, args.nlinesearch)
    s = tree_scale(p, alpha)
    params = tree_add(params, s)

    opt_state = bfgs.update_post(opt_state, alpha)

    state = opt_state, params

    return state, loss

def _gd_init(args, init_params):
    p, r, s = init_params

    params = jnp.log(p), jnp.log(r), jax.scipy.special.logit(s)

    state = (params,)

    def _step(state, f):
        return _gd_step(args, state, f)

    def _params(state):
        (params,) = state
        logits, logr, logs = params
        p, r, s = jax.nn.softmax(logits), jnp.exp(logr), jax.scipy.special.expit(logs)
        return p, r, s

    return state, _step, _params

def _gd_step(args, state, f):
    (params,) = state

    def _f(_):
        logits, logr, logs = _
        p, r, s = jax.nn.softmax(logits), jnp.exp(logr), jax.scipy.special.expit(logs)
        return f((p, r, s))

    loss, g = jax.value_and_grad(_f)(params)
    
    p = tree_scale(g, -1)
    alpha = linesearch.grid_linesearch(params, p, _f, 1.0, args.nlinesearch)
    s = tree_scale(p, alpha)
    params = tree_add(params, s)

    state = (params,)

    return state, loss

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='method', required=True)

    parser_shared = argparse.ArgumentParser(add_help=False)
    parser_shared.add_argument('--k', type=int, default=10)
    parser_shared.add_argument('--nevals', type=int, default=100)
    parser_shared.add_argument('--nlinesearch', type=int, default=1_000)
    parser_shared.add_argument('--data-seed', type=int, default=0)
    parser_shared.add_argument('--init-seed', type=int, default=0)
    parser_shared.add_argument('--dataset', choices=['synthetic'], required=True)

    parser_sngd = subparsers.add_parser('sngd', parents=[parser_shared])
    parser_sngd.set_defaults(init=_sngd_init)

    parser_gd = subparsers.add_parser('gd', parents=[parser_shared])
    parser_gd.set_defaults(init=_gd_init)

    parser_bfgs = subparsers.add_parser('bfgs', parents=[parser_shared])
    parser_bfgs.set_defaults(init=_bfgs_init)

    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.init_seed)

    x = {
        'synthetic': _generate_task(args.data_seed, args.k, 1_000)
    }[args.dataset]

    init_params, rng = rngcall(_init_params, rng, args.k, jnp.max(x)*.1, jnp.max(x)*.9)
    state, step, _ = args.init(args, init_params)

    for i in range(args.nevals+1):
        state, loss = step(state, lambda _: -negbinmix.logprob(_, x).mean(-1))
        print(i, loss)

if __name__ == "__main__":
    main()
