import jax.numpy as jnp
from jax import jit
from jax.experimental.host_callback import call
from jax.lax import while_loop, cond
from jax.random import split, PRNGKey
from scipy.stats import describe
from tensorflow_probability.substrates.jax.mcmc import sample_chain, HamiltonianMonteCarlo, \
    SimpleStepSizeAdaptation, RandomWalkMetropolis
from tqdm.auto import tqdm

from src.ess_solver import ess_solver
from src.resampling import systematic
from src.utils import normalize


def _default_kernel(log_prob_fn):
    hmc = HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        num_leapfrog_steps=10,
        step_size=0.01)

    mcmc = SimpleStepSizeAdaptation(hmc, 50)
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
                            **mcmc_chain_kwargs
                            ):
    key, subkey = split(PRNGKey(seed))
    init_particles = m0(num_particles, subkey)
    init_weights = jnp.full((num_particles,), 1 / num_particles)

    tqdm_message = "Tempering the potential, lambda={:.4f}"
    pbar = tqdm(desc=tqdm_message.format(0.))

    def tqdm_callback(lambda_i):
        pbar.update(1)
        pbar.set_description(tqdm_message.format(float(lambda_i)))

    def loop_cond(carry):
        i, _, lambda_i, *_ = carry
        return jnp.logical_and(lambda_i < max_lambda, i < max_n)

    def body(carry):
        i, particles, lambda_i, weights_i, _, key = carry

        # Get the individual seeds for next iteration, resampling, and MCMC step
        key, subkey1, subkey2 = split(key, 3)

        def run_chain(current_state):
            kernel = make_kernel(jit(lambda x: -lambda_i * potential(x)))

            samples, is_accepted = sample_chain(kernel=kernel,
                                                current_state=current_state,
                                                seed=subkey2,
                                                # trace_fn=lambda _, pkr: pkr.is_accepted,
                                                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                                                **mcmc_chain_kwargs)
            return samples

        def _resample(p):
            # Compute the ancestors
            ancestors = resampling_method(weights_i, subkey1)
            # Select the particles
            p = p[ancestors]

            # Run the monte carlo routine
            p = run_chain(p)[-1]

            return p

        particles = cond(i > 0, _resample, lambda p: p, operand=particles)

        # Solve for the lambda increment
        delta_i = ess_solver(particles, ess_target, potential)

        log_weights = -delta_i * potential(particles)
        call(tqdm_callback, lambda_i)
        return i + 1, particles, lambda_i + delta_i, normalize(log_weights), delta_i, key

    init_carry = (0, init_particles, 0., init_weights, 0., key)
    n, final_particles, lambda_n, final_weights, *_ = while_loop(loop_cond,
                                                                 body,
                                                                 init_carry)
    pbar.close()
    return n, final_particles, lambda_n, final_weights


if __name__ == '__main__':
    from jax.random import normal
    import matplotlib.pyplot as plt

    D = 500


    def potential(x):
        res = -jnp.sum(jnp.sinc(x), -1)
        return res


    def m0(n, key):
        return normal(key, (n, D))


    n, res, lambda_n, final_weights = infinitely_tempered_smc(potential, 1000, 0.5, m0, num_results=50,
                                                              max_lambda=100, max_n=100000)

    print(final_weights)
    print(res.mean(0))
    print(res.std(0))
    print(describe(res, 0))

    plt.hist(res[..., :5].T, bins=100, alpha=0.5)
    plt.show()
