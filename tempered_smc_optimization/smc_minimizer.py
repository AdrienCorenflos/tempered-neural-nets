import jax.numpy as jnp
from jax import jit
from jax.experimental.host_callback import call
from jax.lax import while_loop, cond
from jax.ops import index_update
from jax.random import split, PRNGKey
from tensorflow_probability.substrates.jax.mcmc import sample_chain, HamiltonianMonteCarlo, \
    SimpleStepSizeAdaptation
from tqdm.auto import tqdm

from tempered_smc_optimization.ess_solver import ess_solver
from tempered_smc_optimization.resampling import systematic
from tempered_smc_optimization.utils import normalize


def _default_kernel(log_prob_fn):
    mcmc = HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        num_leapfrog_steps=25,
        step_size=0.01)
    return mcmc


def infinitely_tempered_smc(potential,
                            num_particles,
                            ess_target,
                            m0,
                            make_kernel=_default_kernel,
                            resampling_method=systematic,
                            seed=0,
                            max_lambda=1e2,
                            max_n=1000,
                            record_particles=True,
                            **mcmc_chain_kwargs,
                            ):
    """

    Parameters
    ----------
    potential
    num_particles
    ess_target
    m0
    make_kernel
    resampling_method
    seed
    max_lambda
    max_n
    mcmc_chain_kwargs

    Returns
    -------

    """
    key, subkey1 = split(PRNGKey(seed), 2)
    init_particles = m0(num_particles, subkey1)
    init_weights = jnp.full((num_particles,), 1 / num_particles)

    tqdm_message = "Tempering the potential, lambda={:.6f}, max_lambda={:.1f}"
    pbar = tqdm(desc=tqdm_message.format(0., max_lambda), total=max_n)

    init_lambdas = jnp.zeros(max_n)
    typ = type(init_particles)
    if record_particles:
        all_particles_init = typ(jnp.zeros((max_n, num_particles, *v.shape[1:])) for v in init_particles)
        all_particles_init = typ(index_update(k, 0, v) for k, v in zip(all_particles_init, init_particles))
        means_init = ()
        stds_init = ()
    else:
        all_particles_init = ()
        means_init = typ(jnp.zeros((max_n, *v.shape[1:])) for v in init_particles)
        stds_init = typ(jnp.zeros((max_n, *v.shape[1:])) for v in init_particles)
        means_init = typ(index_update(k, 0, v.mean(0)) for k, v in zip(means_init, init_particles))
        stds_init = typ(index_update(k, 0, v.std(0)) for k, v in zip(stds_init, init_particles))

    def tqdm_callback(lambda_i):
        pbar.update(1)
        pbar.set_description(tqdm_message.format(float(lambda_i), max_lambda))

    def _resample(p_list, w, k):
        # Compute the ancestors
        ancestors = resampling_method(w, k)
        # Select the particles
        return typ(p[ancestors] for p in p_list)

    def loop_cond(carry):
        i, _, lambda_i, *_ = carry
        return jnp.logical_and(lambda_i < max_lambda, i < max_n)

    def body(carry):
        i, particles, lambda_i, weights_i, _, all_particles_i, lambdas, means, stds, key = carry

        # Get the individual seeds for next iteration, resampling, and MCMC step
        key, subkey1, subkey2, subkey3, subkey4 = split(key, 5)

        def run_chain(current_state):
            kernel = make_kernel(jit(lambda *x: -lambda_i * potential(*x, k=subkey3)))
            samples = sample_chain(kernel=kernel,
                                   current_state=current_state,
                                   seed=subkey2,
                                   trace_fn=None,
                                   **mcmc_chain_kwargs)
            samples = typ(s[-1] for s in samples)
            return samples

        particles = cond(i > 0, lambda p: run_chain(p), lambda p: p, operand=particles)
        # Solve for the lambda increment
        delta_i = ess_solver(particles, ess_target, potential, subkey4, num_particles)
        log_weights = -delta_i * potential(*particles, k=subkey4)
        weights = normalize(log_weights)

        lambda_i = lambda_i + delta_i
        call(tqdm_callback, lambda_i - delta_i)

        lambdas = index_update(lambdas, i + 1, lambda_i, True, True)
        particles = _resample(particles, weights, subkey1)
        if record_particles:
            all_particles_i = typ(index_update(k, i + 1, v) for k, v in zip(all_particles_i, particles))
        else:
            means = typ(index_update(k, i + 1, v.mean(0)) for k, v in zip(means, particles))
            stds = typ(index_update(k, i + 1, v.std(0)) for k, v in zip(stds, particles))

        return i + 1, particles, lambda_i, weights, delta_i, all_particles_i, lambdas, means, stds, key

    init_carry = (0, init_particles, 0., init_weights, 0., all_particles_init, init_lambdas, means_init, stds_init, key)
    n, final_particles, lambda_n, final_weights, _, particles_history, final_lambdas, means_history, stds_history, _ = \
        while_loop(
            loop_cond,
            body,
            init_carry)

    pbar.close()
    if record_particles:
        return particles_history, final_lambdas, n
    else:
        return final_particles, final_lambdas, n, means_history, stds_history
