import argparse
import jax
import jax.numpy as jnp
import jax.example_libraries.optimizers
from sngd.distributions.ef import mvn
from sngd.distributions.mcef import mvsn as mcef_mvsn
from sngd.distributions.standard import mvsn
from sngd.util.la import *
from sngd.optimisation.linesearch import backtracking_linesearch
from sngd.util.random import rngcall
from sngd.util.tree import tree_scale, tree_sub
from jax.lax import scan
from jax.random import split

jax.config.update("jax_enable_x64", True)

def _init_params(rng, d):
    xi = jax.random.normal(rng, (d,))*.01
    Omega = jnp.eye(d)*.01
    eta = jax.random.normal(rng, (d,))*.01
    return xi, Omega, eta

def _generate_task(seed, d, n):
    rng = jax.random.PRNGKey(seed)
    key_x, key_y, key_z = split(rng, 3)
    x = jax.random.normal(key_x, (n,d))
    z = jax.random.normal(key_z, (d,))
    y = jax.random.bernoulli(key_y, jax.nn.sigmoid((2*x-1) @ z))
    return x, y

def _sngd_init(args, init_params):
    xi, Omega, eta = init_params

    natparams = mvn.natparams_from_standard((xi, Omega))

    optfuns = jax.example_libraries.optimizers.adam(args.lr_lam)
    opt_init = optfuns[0]
    opt_state = opt_init(eta)
    opt_step = 0

    state = natparams, (opt_state, opt_step)

    def step(state, f):
        return _sngd_step(args, state, optfuns, f)

    def params(state):
        natparams, (opt_state, _) = state
        xi, Omega = mvn.standardparams(natparams)
        eta = optfuns[2](opt_state)
        return xi, Omega, eta

    return state, step, params

def _sngd_step(args, state, optfuns, f):
    natparams, (opt_state, opt_step) = state
    _, opt_update, opt_params = optfuns

    def _f(meanparams, eta):
        meanparams = mvn.symmetrise(meanparams)
        xi, Omega = mvn.standardparams_from_mean(meanparams)
        return f((xi, Omega, eta))
    loss, g = jax.value_and_grad(_f, [0,1])(mvn.meanparams(natparams), opt_params(opt_state))

    if args.backtracking:
        alpha = backtracking_linesearch(
            lambda *_: mvn.innaturaldomain(_[0]), natparams, tree_scale(g[0], -1), args.lr_theta)
    else:
        alpha = args.lr_theta

    natparams = tree_sub(natparams, tree_scale(g[0], alpha))

    opt_state = opt_update(opt_step, g[1], opt_state)

    state = natparams, (opt_state, opt_step + 1)

    return state, loss

def _adam_init(args, init_params):
    xi, Omega, eta = init_params

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
    opt_state = opt_init((xi, W, eta))
    opt_step = 0

    state = opt_state, opt_step

    def step(state, f):
        return _adam_step(args, state, optfuns, f)

    def params(state):
        opt_state = state[0]
        xi, W, eta = optfuns[2](opt_state)
        Omega = _from_W(W)
        return xi, Omega, eta

    return state, step, params

def _adam_step(args, state, optfuns, f):
    opt_state, opt_step = state
    _, opt_update, opt_params = optfuns

    _from_W = {
        'covariance-msqrt': lambda _: transpose(_) @ _,
        'precision-msqrt': lambda _: jnp.linalg.inv(transpose(_) @ _)
    }[args.parameterisation]

    def _f(_):
        xi, W, eta = _
        Omega = _from_W(W)
        return f((xi, Omega, eta))
    loss, g = jax.value_and_grad(_f, 0)(opt_params(opt_state))

    opt_state = opt_update(opt_step, g, opt_state)

    state = opt_state, opt_step + 1

    return state, loss

def _mcef_init(args, init_params):
    natparams = mcef_mvsn.natparams_from_standard(init_params)

    state = natparams, 0

    def step(state, f):
        return _mcef_step(args, state, f)

    def params(state):
        natparams = state[0]
        return mcef_mvsn.standardparams(natparams)

    return state, step, params

def _mcef_step(args, state, f):
    natparams, opt_step = state

    def _f(meanparams):
        params = mcef_mvsn.standardparams(mcef_mvsn.natparams(meanparams))
        return f(params)
    loss, g = jax.value_and_grad(_f, 0)(mcef_mvsn.meanparams(natparams))

    natparams = tree_sub(natparams, tree_scale(g, args.lr))

    state = natparams, opt_step + 1

    return state, loss

def _elbo(rng, n, x, y, regularisation, qparams, nsamples):
    Ez = mvsn.mean(qparams)
    Vz = mvsn.var(qparams)
    Ezz = diagv(Vz) + Ez*Ez
    Elogp = jnp.sum(-.5*Ezz*regularisation - .5*jnp.log(2*jnp.pi/regularisation), 0)
    Elogq = -mvsn.entropy(qparams)
    z = mvsn.sample(rng, qparams, nsamples)
    logits = jnp.sum(z[:,None] * x[None], -1)
    Ef = n*jax.nn.log_sigmoid((y*2-1)*logits).mean([0,1])
    return Ef + Elogp - Elogq

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='method', required=True)

    parser_shared = argparse.ArgumentParser(add_help=False)
    parser_shared.add_argument('--nsteps-per-eval', type=int, default=10)
    parser_shared.add_argument('--nevals', type=int, default=100)
    parser_shared.add_argument('--nsamples-train', type=int, default=10)
    parser_shared.add_argument('--nsamples-eval', type=int, default=100)
    parser_shared.add_argument('--data-seed', type=int, default=0)
    parser_shared.add_argument('--init-seed', type=int, default=0)
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

    parser_mcef = subparsers.add_parser('mcef', parents=[parser_shared])
    parser_mcef.add_argument('--lr', type=float, required=True)
    parser_mcef.set_defaults(init=_mcef_init)

    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.init_seed)

    regularisation = 1.0
    x, y = {
        'synthetic': _generate_task(args.data_seed, d=100, n=100)
    }[args.dataset]
    n, d = x.shape

    init_params, rng = rngcall(_init_params, rng, d)
    state, step, params = args.init(args, init_params)

    def train(key, state):
        def _step(s, k):
            return step(s, lambda _: -_elbo(k, n, x, y, regularisation, _, args.nsamples_train))
        return scan(_step, state, split(key, args.nsteps_per_eval))
    
    def eval(key, state):
        return -_elbo(key, n, x, y, regularisation, params(state), args.nsamples_eval)

    for i in range(args.nevals):
        key_eval, key_train, rng = split(rng, 3)
        evalloss = eval(key_eval, state)
        state, trainloss = train(key_train, state)
        print(i, trainloss.mean(), evalloss)

if __name__ == "__main__":
    main()
