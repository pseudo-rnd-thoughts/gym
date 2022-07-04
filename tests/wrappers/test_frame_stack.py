import numpy as np
import pytest

import gym
from gym import spaces
from gym.wrappers import FrameStack

try:
    import lz4
except ImportError:
    lz4 = None


# todo - These tests should be run with Atari environments however due to current implementations, this is not possible
@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1", "CarRacing-v1"])
@pytest.mark.parametrize("num_stack", [2, 3, 4])
@pytest.mark.parametrize("lz4_compress", [lz4 is not None, False])
def test_frame_stack(env_id, num_stack, lz4_compress):
    env = gym.make(env_id, disable_env_checker=True)
    obs_space = env.observation_space
    env = FrameStack(env, num_stack, lz4_compress)

    assert isinstance(obs_space, spaces.Box)
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (num_stack,) + obs_space.shape
    assert env.observation_space.dtype == obs_space.dtype
    assert np.all(env.observation_space.low[i] == obs_space.low for i in range(num_stack))

    dup = gym.make(env_id, disable_env_checker=True)

    obs = env.reset(seed=0)
    dup_obs = dup.reset(seed=0)
    assert np.allclose(obs[-1], dup_obs)

    for _ in range(num_stack**2):
        action = env.action_space.sample()
        dup_obs, _, _, _ = dup.step(action)
        obs, _, _, _ = env.step(action)
        assert np.allclose(obs[-1], dup_obs)

    assert len(obs) == num_stack
