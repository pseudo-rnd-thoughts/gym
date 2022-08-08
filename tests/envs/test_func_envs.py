import pytest

from gym.envs.jax_env import jax_env_checker
from gym.envs.phys2d import FunctionalCartPole, FunctionalPendulum
from gym.functional import FunctionalEnv


@pytest.mark.parametrize("func_env", [FunctionalCartPole(), FunctionalPendulum()])
def test_jax_func_envs(func_env: FunctionalEnv):
    jax_env_checker(func_env)
