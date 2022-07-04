import numpy as np
import pytest

import gym
from gym import spaces
from gym.wrappers import GrayScaleObservation

pytest.importorskip("cv2")


# todo - These tests should be run with Atari environments however due to current implementations, this is not possible
@pytest.mark.parametrize("env_id", ["CarRacing-v1"])
@pytest.mark.parametrize("keep_dim", [True, False])
def test_gray_scale_observation(env_id, keep_dim):
    rgb_env = gym.make(env_id, disable_env_checker=True)
    assert len(rgb_env.observation_space.shape) == 3
    assert rgb_env.observation_space.shape[-1] == 3
    assert rgb_env.observation_space.dtype == np.uint8

    grayscale_env = GrayScaleObservation(rgb_env, keep_dim=keep_dim)
    assert grayscale_env.action_space is rgb_env.action_space
    assert isinstance(grayscale_env.observation_space, spaces.Box)
    if keep_dim:
        assert len(grayscale_env.observation_space.shape) == 3
        assert grayscale_env.observation_space.shape[-1] == 1
    else:
        assert len(grayscale_env.observation_space.shape) == 2
    assert grayscale_env.observation_space.shape[:2] == rgb_env.observation_space.shape[:2]

    grayscale_obs = grayscale_env.reset(seed=123)
    assert grayscale_obs in grayscale_env.observation_space

    grayscale_obs, _, _, _ = grayscale_env.step(grayscale_env.action_space.sample())
    assert grayscale_obs in grayscale_env.observation_space
