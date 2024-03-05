import argparse
import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers
from sngd.distributions.ef import mvn
from sngd.distributions.standard import tcopula
from sngd.util.la import *
from sngd.optimisation.linesearch import backtracking_linesearch
from sngd.util.random import rngcall
from sngd.util.stats import cov2corr
from sngd.util.tree import tree_scale, tree_sub
from jax.lax import scan
from jax.random import split

jax.config.update("jax_enable_x64", True)

def _init_params(rng, d):
    W = jax.random.normal(rng, ((d,d)))*.01
    R = cov2corr(jnp.eye(d) + W.T @ W)
    nu = jnp.array(50.0)
    return R, nu

def _generate_task(seed, d, n):
    rng = jax.random.PRNGKey(seed)
    W, rng = rngcall(jax.random.normal, rng, ((d,d)))
    R = cov2corr((W.T @ W)/d + 1e-4*jnp.eye(d))
    nu = jnp.array(5.0)
    return tcopula.sample(rng, (R, nu), n)

def _sngd_init(args, init_params):
    R, nu = init_params

    meanparams = mvn.meanparams_from_standard(
        (jnp.zeros(R.shape[-1]), R))

    optfuns = jax.example_libraries.optimizers.adam(args.lr_lam)
    opt_init = optfuns[0]
    opt_state = opt_init(jnp.log(nu-args.min_nu))
    opt_step = 0

    state = meanparams, (opt_state, opt_step)

    def step(state, f):
        return _sngd_step(args, state, optfuns, f)

    def params(state):
        meanparams, (opt_state, _) = state
        R = cov2corr(mvn.var(mvn.natparams(meanparams)))
        lognu = optfuns[2](opt_state)
        return R, jnp.exp(lognu)+args.min_nu

    return state, step, params

def _sngd_step(args, state, optfuns, f):
    meanparams, (opt_state, opt_step) = state
    _, opt_update, opt_params = optfuns

    def _f(natparams, lognu):
        natparams = mvn.symmetrise(natparams)
        R = cov2corr(mvn.var(natparams))
        return f((R, jnp.exp(lognu)+args.min_nu))
    loss, g = jax.value_and_grad(_f, [0,1])(mvn.natparams(meanparams), opt_params(opt_state))

    if args.backtracking:
        alpha = backtracking_linesearch(
            lambda *_: mvn.inmeandomain(_[0]), meanparams, tree_scale(g[0], -1), args.lr_theta)
    else:
        alpha = args.lr_theta

    meanparams = tree_sub(meanparams, tree_scale(g[0], alpha))

    opt_state = opt_update(opt_step, g[1], opt_state)

    state = meanparams, (opt_state, opt_step + 1)

    return state, loss

def _adam_init(args, init_params):

    R, nu = init_params

    _to_W = {
        'covariance-msqrt': lambda _: jnp.linalg.cholesky(_),
        'precision-msqrt': lambda _: jnp.linalg.cholesky(jnp.linalg.inv(_))
    }[args.parameterisation]

    _from_W = {
        'covariance-msqrt': lambda _: transpose(_) @ _,
        'precision-msqrt': lambda _: jnp.linalg.inv(transpose(_) @ _)
    }[args.parameterisation]

    W = _to_W(R)

    optfuns = jax.example_libraries.optimizers.adam(args.lr)
    opt_init = optfuns[0]
    opt_state = opt_init((W, jnp.log(nu-args.min_nu)))
    opt_step = 0

    state = opt_state, opt_step

    def step(state, f):
        return _adam_step(args, state, optfuns, f)

    def params(state):
        opt_state = state[0]
        W, lognu = optfuns[2](opt_state)
        R = cov2corr(_from_W(W))
        return R, jnp.exp(lognu) + args.min_nu

    return state, step, params

def _adam_step(args, state, optfuns, f):
    opt_state, opt_step = state
    _, opt_update, opt_params = optfuns

    _from_W = {
        'covariance-msqrt': lambda _: transpose(_) @ _,
        'precision-msqrt': lambda _: jnp.linalg.inv(transpose(_) @ _)
    }[args.parameterisation]

    def _f(_):
        W, lognu = _
        R = cov2corr(_from_W(W))
        return f((R, jnp.exp(lognu)+args.min_nu))
    loss, g = jax.value_and_grad(_f, 0)(opt_params(opt_state))

    opt_state = opt_update(opt_step, g, opt_state)

    state = opt_state, opt_step + 1

    return state, loss

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='method', required=True)

    parser_shared = argparse.ArgumentParser(add_help=False)
    parser_shared.add_argument('--nsteps-per-eval', type=int, default=10)
    parser_shared.add_argument('--nevals', type=int, default=100)
    parser_shared.add_argument('--ntest', type=int, default=100)
    parser_shared.add_argument('--data-seed', type=int, default=0)
    parser_shared.add_argument('--init-seed', type=int, default=0)
    parser_shared.add_argument('--batch-size', type=int, default=100)
    parser_shared.add_argument('--min-nu', type=float, default=2.0)
    parser_shared.add_argument('--dataset', choices=['synthetic'], required=True)

    parser_sngd = subparsers.add_parser('sngd', parents=[parser_shared])
    parser_sngd.add_argument('--lr-theta', type=float, required=True)
    parser_sngd.add_argument('--lr-lam', type=float, required=True)
    parser_sngd.add_argument('--backtracking', dest='backtracking', default=False, action='store_true')
    parser_sngd.set_defaults(init=_sngd_init)

    parser_adam = subparsers.add_parser('adam', parents=[parser_shared])
    parser_adam.add_argument('--lr', type=float, required=True)
    parser_adam.add_argument('--parameterisation', choices=['covariance-msqrt', 'precision-msqrt'], required=True)
    parser_adam.set_defaults(init=_adam_init)

    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.init_seed)

    x = {
        'synthetic': _generate_task(args.data_seed, d=100, n=1_000)
    }[args.dataset]

    x, x_test = jnp.split(x, [x.shape[0]-args.ntest], 0)
    n, d = x.shape

    init_params, rng = rngcall(_init_params, rng, d)
    state, step, params = args.init(args, init_params)

    def get_batch(key):
        if args.batch_size is None:
            return x
        else:
            return x[jax.random.choice(key, n, (args.batch_size,), replace=True)] 

    def train(key, state):
        def _step(s, k):
            return step(s, lambda _: -tcopula.logprob(_, get_batch(k)).mean(0))
        return scan(_step, state, split(key, args.nsteps_per_eval))
    
    def eval(state):
        return -tcopula.logprob(params(state), x_test).mean(0)

    for i in range(args.nevals):
        key_train, rng = split(rng)
        evalloss = eval(state)
        state, trainloss = train(key_train, state)
        print(i, trainloss.mean(), evalloss)

if __name__ == "__main__":
    main()
