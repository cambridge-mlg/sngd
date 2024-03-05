import argparse
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import jax.example_libraries.optimizers
from sngd.distributions.ef import mvn
from sngd.distributions.standard import mvst
from sngd.util.la import *
from sngd.optimisation.linesearch import backtracking_linesearch
from sngd.util.random import rngcall
from sngd.util.tree import tree_scale, tree_sub
from jax.lax import scan
from jax.random import split

jax.config.update("jax_enable_x64", True)

def _init_params(rng, d):
    xi = jax.random.normal(rng, (d,))*.01
    Omega = jnp.eye(d)
    eta = jax.random.normal(rng, (d,))*.01
    nu = jnp.array(50.)
    return xi, Omega, eta, nu

def _generate_task(seed, d, n):
    rng = jax.random.PRNGKey(seed)
    W, rng = rngcall(jax.random.normal, rng, ((d,d)))
    xi, rng = rngcall(jax.random.normal, rng, (d,))
    Omega = (W.T @ W)/d + 1e-4*jnp.eye(d)
    eta, rng = rngcall(jax.random.normal, rng, (d,))
    nu = jnp.array(10.)
    return mvst.sample(rng, (xi, Omega, eta, nu), n)

def _sngd_init(args, init_params):
    xi, Omega, eta, nu = init_params

    meanparams = mvn.meanparams(
        mvn.natparams_from_standard((xi, Omega)))

    optfuns = jax.example_libraries.optimizers.adam(args.lr_lam)
    opt_init = optfuns[0]
    opt_state = opt_init((eta, jnp.log(nu-args.min_nu)))
    opt_step = 0

    state = meanparams, (opt_state, opt_step)

    def step(state, f):
        return _sngd_step(args, state, optfuns, f)

    def params(state):
        meanparams, (opt_state, _) = state
        xi, Omega = mvn.standardparams(mvn.natparams(meanparams))
        eta, lognu = optfuns[2](opt_state)
        return xi, Omega, eta, jnp.exp(lognu)+args.min_nu

    return state, step, params

def _sngd_step(args, state, optfuns, f):
    meanparams, (opt_state, opt_step) = state
    _, opt_update, opt_params = optfuns

    def _f(natparams, eta, lognu):
        natparams = mvn.symmetrise(natparams)
        xi, Omega = mvn.standardparams(natparams)
        return f((xi, Omega, eta, jnp.exp(lognu)+args.min_nu))
    loss, g = jax.value_and_grad(_f, [0,1,2])(mvn.natparams(meanparams), *opt_params(opt_state))

    if args.backtracking:
        alpha = backtracking_linesearch(
            lambda *_: mvn.inmeandomain(_[0]), meanparams, tree_scale(g[0], -1), args.lr_theta)
    else:
        alpha = args.lr_theta

    meanparams = tree_sub(meanparams, tree_scale(g[0], alpha))

    opt_state = opt_update(opt_step, g[1:], opt_state)

    state = meanparams, (opt_state, opt_step + 1)

    return state, loss

def _adam_init(args, init_params):
    xi, Omega, eta, nu = init_params

    _to_W = {
        'covariance-msqrt': lambda _: jnp.linalg.cholesky(_),
        'precision-msqrt': lambda _: jnp.linalg.cholesky(jnp.linalg.inv(_))
    }[args.parameterisation]

    _from_W = {
        'covariance-msqrt': lambda _: transpose(_) @ _,
        'precision-msqrt': lambda _: jnp.linalg.inv(transpose(_) @ _)
    }[args.parameterisation]

    W = _to_W(Omega)

    optfuns = jax.example_libraries.optimizers.adam(args.lr)
    opt_init = optfuns[0]
    opt_state = opt_init((xi, W, eta, jnp.log(nu-args.min_nu)))
    opt_step = 0

    state = opt_state, opt_step

    def step(state, f):
        return _adam_step(args, state, optfuns, f)

    def params(state):
        opt_state = state[0]
        xi, W, eta, lognu = optfuns[2](opt_state)
        Omega = _from_W(W)
        return xi, Omega, eta, jnp.exp(lognu) + args.min_nu

    return state, step, params

def _adam_step(args, state, optfuns, f):
    opt_state, opt_step = state
    _, opt_update, opt_params = optfuns

    _from_W = {
        'covariance-msqrt': lambda _: transpose(_) @ _,
        'precision-msqrt': lambda _: jnp.linalg.inv(transpose(_) @ _)
    }[args.parameterisation]

    def _f(_):
        xi, W, eta, lognu = _
        Omega = _from_W(W)
        return f((xi, Omega, eta, jnp.exp(lognu)+args.min_nu))
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
    parser_shared.add_argument('--ntest', type=int, default=1_000)
    parser_shared.add_argument('--data-seed', type=int, default=0)
    parser_shared.add_argument('--init-seed', type=int, default=0)
    parser_shared.add_argument('--batch-size', type=int, default=100)
    parser_shared.add_argument('--dataset', choices=['synthetic'], required=True)
    parser_shared.add_argument('--min-nu', type=float, default=2.0)

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
        'synthetic': _generate_task(args.data_seed, d=1_000, n=10_000)
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
            return step(s, lambda _: -mvst.logprob(_, get_batch(k)).mean(0), )
        return scan(_step, state, split(key, args.nsteps_per_eval))
    
    def eval(state):
        return -mvst.logprob(params(state), x_test).mean(0)

    for i in range(args.nevals):
        key_train, rng = split(rng)
        evalloss = eval(state)
        state, trainloss = train(key_train, state)
        print(i, trainloss.mean(), evalloss)

if __name__ == "__main__":
    main()
