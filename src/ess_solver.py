import jax.numpy as jnp
from jax.lax import while_loop
from tensorflow_probability.substrates.jax.math import find_root_chandrupatla

from src.utils import ess


def ess_solver(particles, ess_target, potential):
    """

    Parameters
    ----------
    particles
    ess_target
    potential

    Returns
    -------

    Examples
    --------
    >>> potential = lambda x: x ** 2
    >>> particles = jnp.linspace(-1., 1., 5000)
    >>> ess_solver(particles, 0.5, potential)

    """
    potential_val = potential(particles)
    n = particles.shape[0]

    def fun_to_minimize(delta):
        ess_val = ess(-delta * potential_val)
        return ess_val - n * ess_target

    # find sign inversions
    b0 = 1e-8

    def cond(carry):
        _, val = carry
        return val > 0.

    def body(carry):
        bi, _ = carry
        bi = 2 * bi
        val = fun_to_minimize(bi)
        return bi, val

    b, _ = while_loop(cond, body, (b0, 1.))
    delta = find_root_chandrupatla(fun_to_minimize, 0., b, value_tolerance=1e-2)
    return delta.estimated_root
    # try:
    #     return call(host_call_function, (),
    #                 result_shape=ShapeDtypeStruct((), potential_val.dtype))
    # except ValueError:
    #     return max_delta
