import pytest
import numpy as np

from src.ess_solver import ess_solver


@pytest.mark.parametrize("ess_target", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test(ess_target, seed):
    rng = np.random.RandomState(seed)

    def potential(x):
        return x ** 2

    particles = rng.uniform(-1, 1, (5000,))
    res = ess_solver(particles, ess_target, potential, 1.)
    print(res)
