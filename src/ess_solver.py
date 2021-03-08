import jax.numpy as jnp
from jax import jit, ShapeDtypeStruct

from src.utils import ess, newton
from jax.experimental.host_callback import call, id_print
from jax.scipy.optimize import minimize


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
    >>> ess_solver(particles, 0., potential)
    """
    potential_val = potential(particles)
    n = particles.shape[0]

    def fun_to_minimize(delta):
        delta_neg_penalty = jnp.maximum(-delta, 0) ** 2
        ess_val = ess(-delta * potential_val)
        return (ess_val - ess_target * n) ** 2 + delta_neg_penalty

    delta = newton(fun_to_minimize, 10.)
    return delta
    # try:
    #     return call(host_call_function, (),
    #                 result_shape=ShapeDtypeStruct((), potential_val.dtype))
    # except ValueError:
    #     return max_delta
