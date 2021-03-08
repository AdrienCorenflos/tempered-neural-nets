from jax.random import uniform
import jax.numpy as jnp


def systematic(weights, key):
    """
    Systematic resampling method

    Parameters
    ----------
    weights: array_like
        Weights to resample from
    key: PRNGKey
        The random key used

    Returns
    -------
    idx: array_like
        The indices of the resampled particles
    """
    n_samples = weights.shape[0]
    cumsum = jnp.cumsum(weights)
    u = uniform(key)
    aux = (u + jnp.arange(n_samples)) / n_samples
    return jnp.searchsorted(cumsum, aux)

def metropolis(log_weights, key, b):
    raise NotImplementedError("TODO")