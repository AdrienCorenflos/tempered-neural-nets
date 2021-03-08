import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.host_callback import id_print
from jax.random import split, PRNGKey
from jax.lax import while_loop

from src.ess_solver import ess_solver
from src.resampling import systematic
from src.utils import normalize
import enum
from tensorflow_probability.substrates.jax.mcmc import sample_chain, RandomWalkMetropolis, HamiltonianMonteCarlo


def _default_kernel(log_prob_fn):
    return HamiltonianMonteCarlo(log_prob_fn, step_size=1e-3, num_leapfrog_steps=10)


def infinitely_tempered_smc(potential,
                            num_particles,
                            ess_target,
                            m0,
                            make_kernel=_default_kernel,
                            resampling_method=systematic,
                            seed=0,
                            max_delta=1e2,
                            max_n=100,
                            **mcmc_chain_kwargs
                            ):
    key, subkey = split(PRNGKey(seed))
    init_particles = m0(num_particles, subkey)

    def cond(carry):
        i, *_, delta_i, _ = carry
        return jnp.logical_and(delta_i < max_delta, i < max_n)

    def body(carry):
        i, particles, lambda_i, mean_accepted, _, key = carry

        # Get the individual seeds for next iteration, resampling, and MCMC step
        key, subkey1, subkey2 = split(key, 3)

        # Solve for the lambda increment
        delta_i = ess_solver(particles, ess_target, potential)

        # Compute the ancestors
        log_weights = -delta_i * potential(particles)
        weights = normalize(log_weights)

        ancestors = resampling_method(weights, subkey1)

        # Select the particles
        particles = particles[ancestors]
        # Run a (few) MCMC kernel step on the particles
        kernel = make_kernel(jit(lambda x: -delta_i * potential(x)))

        def run_chain(current_state):
            samples, is_accepted = sample_chain(kernel=kernel,
                                                current_state=current_state,
                                                seed=subkey2,
                                                trace_fn=lambda _, pkr: pkr.is_accepted,
                                                **mcmc_chain_kwargs)
            return samples, jnp.mean(is_accepted, 0)

        chain_samples, step_mean_accepted = run_chain(particles)

        particles = chain_samples[-1]
        mean_accepted = (i * mean_accepted + step_mean_accepted)

        return i + 1, particles, lambda_i + delta_i, mean_accepted, delta_i, key

    init_carry = (0, init_particles, 0., jnp.zeros((num_particles,)), 0., key)
    n, final_particles, lambda_n, mean_accepted_n, *_ = while_loop(cond,
                                                                   body,
                                                                   init_carry)
    return n, final_particles, lambda_n, mean_accepted_n


if __name__ == '__main__':
    from jax.random import normal
    import matplotlib.pyplot as plt

    D = 100


    def potential(x):
        return jnp.sum((x - 1) ** 2, -1)


    def m0(n, key):
        return 10. * normal(key, (n, D))


    n, res, lambda_n, mean_accepted_n = infinitely_tempered_smc(potential, 100, 0.75, m0, num_results=50,
                                                                max_delta=1e3)
    print(res.mean(0))
    print(res.std(0))
