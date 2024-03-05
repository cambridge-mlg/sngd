import argparse
import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers
import scipy.stats
import numpy.random as npr
from sngd.distributions.ef import gamma
from sngd.optimisation import bfgs, linesearch
from sngd.util.la import *
from sngd.util.random import rngcall
from sngd.util.tree import tree_add, tree_scale
from jax import grad
from jax.random import split
from jax.scipy.stats import nbinom

jax.config.update("jax_enable_x64", True)

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

def _gammamean_to_negbin(_):
    return  _gamma_to_negbin(
        gamma.standardparams(
            gamma.natparams(_)))

def _negbin_to_gammamean(_):
    alpha, beta = _negbin_to_gamma(_)
    return gamma.meanparams(gamma.natparams_from_standard((alpha, beta)))

def _init_params(key):
    key_r, key_p = split(key)
    r = gamma.sample(key_r, gamma.natparams_from_meanvar(jnp.array(5.0), jnp.array(2.0**2)))
    p = jax.random.uniform(key_p, minval=.05, maxval=.95)
    return r, p

def _generate_task(seed, n):
    m = 50
    v = m + (m**2)*.2
    s = m/v
    r = m*s/(1-s)
    return scipy.stats.nbinom.rvs(r, s, size=n, random_state=npr.RandomState(seed))

def _sngd_init(args, init_params):
    r, p = init_params

    params = _negbin_to_gammamean((r, p))
    state = params, 0

    def _step(state, f):
        return _sngd_step(state, args, f)

    def _params(state):
        params = state[0]
        return _gammamean_to_negbin(params)
    
    return state, _step, _params

def _sngd_step(state, args, f):
    params, opt_step = state

    def _f_natparams(natparams):
        alpha, beta = gamma.standardparams(natparams)
        return f(_gamma_to_negbin((alpha, beta)))
    loss, g = jax.value_and_grad(_f_natparams)(gamma.natparams(params))

    _f_meanparams = lambda _: _f_natparams(gamma.natparams(_))
    p = tree_scale(g, -1)
    alpha = linesearch.grid_linesearch(params, p, _f_meanparams, 1.0, args.nlinesearch)
    params = tree_add(params, tree_scale(p, alpha))

    state = params, opt_step + 1

    return state, loss

def _bfgs_init(args, init_params):
    r, p = init_params

    params = {
        'standard': (jnp.log(r), jax.scipy.special.logit(p)),
        'gammamean': _negbin_to_gammamean((r, p))
    }[args.parameterisation]

    _to_standard = {
        'gammamean': _gammamean_to_negbin,
        'standard': lambda _: (jnp.exp(_[0]), jax.scipy.special.expit(_[1]))
    }[args.parameterisation]

    state = bfgs.init(params), params

    def _step(state, f):
        return _bfgs_step(state, f, _to_standard)

    def _params(state):
        return _to_standard(state[1])

    return state, _step, _params

def _bfgs_step(state, f, _to_standard):
    opt_state, params = state

    def _f(_):
        r, p = _to_standard(_)
        return f((r, p))

    loss, g = jax.value_and_grad(_f)(params)
    opt_state = bfgs.update_pre(opt_state, g)
    p = bfgs.search_direction(opt_state)
    
    alpha = linesearch.grid_linesearch(params, p, _f, 1.0, 1_000)
    s = tree_scale(p, alpha)
    params = tree_add(params, s)

    opt_state = bfgs.update_post(opt_state, alpha)

    state = opt_state, params

    return state, loss

def _gd_init(args, init_params):
    r, p = init_params

    params = {
        'gammamean': _negbin_to_gammamean((r, p)),
        'standard': (jnp.log(r), jax.scipy.special.logit(p))
    }[args.parameterisation]

    _to_standard = {
        'gammamean': _gammamean_to_negbin,
        'standard': lambda _: (jnp.exp(_[0]), jax.scipy.special.expit(_[1]))
    }[args.parameterisation]

    state = params

    def _step(state, f):
        return _gd_step(state, args, _to_standard, f)

    def _params(state):
        params = state
        return _to_standard(params)
    
    return state, _step, _params

def _gd_step(state, args, _to_standard, f):
    params = state

    def _f(_):
        return f(_to_standard(_))
    loss, g = jax.value_and_grad(_f)(params)

    p = tree_scale(g, -1)
    alpha = linesearch.grid_linesearch(params, p, _f, 1.0, args.nlinesearch)
    params = tree_add(params, tree_scale(p, alpha))

    state = params

    return state, loss

def _ngd_init(args, init_params):
    r, p = init_params

    params = {
        'standard': (jnp.log(r), jax.scipy.special.logit(p)),
        'gammamean': _negbin_to_gammamean((r, p))
    }[args.parameterisation]

    _to_standard = {
        'gammamean': _gammamean_to_negbin,
        'standard': lambda _: (jnp.exp(_[0]), jax.scipy.special.expit(_[1]))
    }[args.parameterisation]

    state = params

    def _step(state, f):
        return _ngd_step(state, args, _to_standard, f)

    def _params(state):
        params = state
        return _to_standard(params)
    
    return state, _step, _params

def _ngd_step(state, args, _to_standard, f):
    params = state
    
    def _f(_):
        return f(_to_standard(_))
    loss, g = jax.value_and_grad(_f)(params)

    xsamp = scipy.stats.nbinom.rvs(*_to_standard(params), size=args.nsamples)
    gs = vmap(
            lambda x: grad(lambda _: nbinom.logpmf(x, *_to_standard(tuple(_))))(jnp.array(params))
        )(xsamp)
    
    F = jnp.mean(outer(gs, gs), 0)
    ng = tuple(jnp.linalg.solve(F, jnp.array(g)))

    p = tree_scale(ng, -1)
    alpha = linesearch.grid_linesearch(params, p, _f, 1.0, args.nlinesearch)
    params = tree_add(params, tree_scale(p, alpha))

    state = params

    return state, loss

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='method', required=True)

    parser_shared = argparse.ArgumentParser(add_help=False)
    parser_shared.add_argument('--nevals', type=int, default=100)
    parser_shared.add_argument('--nlinesearch', type=int, default=1_000)
    parser_shared.add_argument('--data-seed', type=int, default=0)
    parser_shared.add_argument('--init-seed', type=int, default=0)
    parser_shared.add_argument('--dataset', choices=['synthetic'], required=True)

    parser_sngd = subparsers.add_parser('sngd', parents=[parser_shared])
    parser_sngd.set_defaults(init=_sngd_init)

    parser_bfgs = subparsers.add_parser('bfgs', parents=[parser_shared])
    parser_bfgs.add_argument('--parameterisation', choices=['gammamean', 'standard'], required=True)
    parser_bfgs.set_defaults(init=_bfgs_init)

    parser_gd = subparsers.add_parser('gd', parents=[parser_shared])
    parser_gd.add_argument('--parameterisation', choices=['gammamean', 'standard'], required=True)
    parser_gd.set_defaults(init=_gd_init)

    parser_ngd = subparsers.add_parser('ngd', parents=[parser_shared])
    parser_ngd.add_argument('--parameterisation', choices=['gammamean', 'standard'], required=True)
    parser_ngd.add_argument('--nsamples', type=int, default=100_000)
    parser_ngd.set_defaults(init=_ngd_init)

    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.init_seed)

    x = {
        'synthetic': _generate_task(args.data_seed, n=1_000)
    }[args.dataset]

    init_params, rng = rngcall(_init_params, rng)
    state, step, _ = args.init(args, init_params)

    for i in range(args.nevals+1):
        state, loss = step(state, lambda _: -nbinom.logpmf(x, *_).mean(-1))
        print(i, loss)

if __name__ == "__main__":
    main()
