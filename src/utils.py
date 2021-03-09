import numpy as np
from jax import numpy as jnp, jit, grad
from jax.experimental.host_callback import id_print
from jax.lax import while_loop


def ess(log_weights):
    """ Compute the effective sample size from the log-weights in a numerically stable way

    Parameters
    ----------
    w: np.ndarray
        log-weights of the sample

    Returns
    -------
    ess: float
        the effective sample size
    """
    w: jnp.ndarray = jnp.exp(log_weights - jnp.max(log_weights))
    return w.sum() ** 2 / jnp.square(w).sum()


def newton(fun, x0, ftol=1e-5, gtol=1e-3, max_iter=100):
    """
    A simple Newton-Raphston method.

    Parameters
    ----------
    fun: callable
        The function we want the root of
    x0: float
        starting point
    ftol: float
        function tolerance
    gtol: float
        gradient tolerance
    max_iter: int
        Maximum number of iterations

    Returns
    -------
    root: float
        The root of the function

    Examples
    --------
    >>> f = lambda x: x**2
    >>> float(newton(f, 1.))
    0.0009765625
    """

    # Wrap the objective function to produce scalar outputs.
    grad_f = jit(grad(fun, 0))

    f0 = fun(x0)
    g0 = grad_f(x0)

    def cond(carry):
        i, _xi, gi, f_diff, _ = carry
        return jnp.all(jnp.array([i < max_iter, jnp.abs(gi) > gtol, jnp.abs(f_diff) > ftol]))

    def body(carry):
        i, xi, gi, _, fi = carry

        xip1 = xi - fi / gi

        fip1 = fun(xip1)
        gip1 = grad_f(xip1)

        return i + 1, xip1, gip1, fip1 - fi, fip1

    # Minimize with scipy
    n, sol, *_ = while_loop(cond, body, (1, x0, g0, 2 * ftol, f0))
    return sol


def normalize(log_weights):
    """
    Normalize log-weights into weights
    
    Parameters
    ----------
    log_weights

    Returns
    -------

    """
    w = jnp.exp(log_weights - jnp.max(log_weights))
    return w / w.sum()
