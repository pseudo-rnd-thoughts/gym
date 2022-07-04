import pytest

import gym
import numpy as np
from gym.spaces import Box, Discrete, Dict, Tuple
from gym.wrappers import grayscale_observations_v0
from tests.dev_wrappers.test_lambda_observations.mock_data_observation import (
    DISCRETE_VALUE,
    NUM_ENVS,
    NEW_BOX_DIM,
    NEW_BOX_DIM_IMPOSSIBLE,
    NUM_STEPS,
    SEED,
    TESTING_BOX_OBSERVATION_SPACE,
    TESTING_DICT_OBSERVATION_SPACE,
    TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE,
    TESTING_NESTED_DICT_ACTION_SPACE,
    TESTING_TUPLE_OBSERVATION_SPACE,
)
from tests.dev_wrappers.utils import TestingEnv


@pytest.mark.parametrize(("env"), [gym.make("CarRacingDiscrete-v1")])
def test_grayscale_observation_v0(env):
    """Test correct transformation of observation in grayscale."""
    wrapped_env = grayscale_observations_v0(env)
    obs, _, _, _ = wrapped_env.step(DISCRETE_VALUE)

    assert len(obs.shape) == 2 # height and width. No more color dim


@pytest.mark.parametrize(("env"), [gym.vector.make("CarRacingDiscrete-v1", num_envs=NUM_ENVS)])
def test_grayscale_observation_v0_vectorenv(env):
    """Test correct transformation of observation in grayscale."""
    wrapped_env = grayscale_observations_v0(env)
    obs, _, _, _ = wrapped_env.step([DISCRETE_VALUE] * NUM_ENVS)

    assert len(obs.shape) == 3 # height and width. No more color dim
    assert obs.shape[0] == NUM_ENVS
    

@pytest.mark.parametrize(
    ("env", "args",),
    [
        (
            TestingEnv(
            observation_space=Dict(
                obs=Box(0, 255, (96, 96, 3), np.uint8),
                time=Discrete(DISCRETE_VALUE)
                )
            ),
            {"obs": True, "time": False}
        ),
    ]
    )
def test_grayscale_observation_v0_vectorenv(env, args):
    """Test correct transformation of observation in grayscale."""
    wrapped_env = grayscale_observations_v0(env, args)

    assert len(wrapped_env.observation_space["obs"].shape) == 2
    assert wrapped_env.observation_space["time"] == Discrete(DISCRETE_VALUE)

    obs, _, _, _ = wrapped_env.step({"obs": 0})

    assert len(obs["obs"].shape) == 2
