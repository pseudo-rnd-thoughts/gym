import pytest

import gym
import numpy as np
from functools import reduce
import operator as op
from gym.spaces import Box, Discrete, Dict, Tuple
from gym.wrappers import flatten_observations_v0
from tests.dev_wrappers.test_lambda_observations.mock_data_observation import (
    DISCRETE_VALUE,
    NUM_ENVS,
    NEW_BOX_DIM,
    NEW_BOX_DIM_IMPOSSIBLE,
    FLATTENEND_DICT_SIZE,
    NUM_STEPS,
    SEED,
    TESTING_BOX_OBSERVATION_SPACE,
    TESTING_DICT_OBSERVATION_SPACE,
    TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE,
    TESTING_NESTED_DICT_ACTION_SPACE,
    TESTING_TUPLE_OBSERVATION_SPACE,
)
from tests.dev_wrappers.utils import TestingEnv


@pytest.mark.parametrize(("env"), [gym.make("CarRacingDiscrete-v1"), ])
def test_flatten_observation_v0(env):
    """Test correct flattening of observation space."""
    flattened_shape = reduce(op.mul, env.observation_space.shape, 1)
    wrapped_env = flatten_observations_v0(env)
    obs, _, _, _ = wrapped_env.step(DISCRETE_VALUE)

    assert wrapped_env.observation_space.shape[0] == flattened_shape
    assert obs.shape[0] == flattened_shape


@pytest.mark.parametrize(
    ("env", "flattened_size"),
    [
        (
            TestingEnv(observation_space=TESTING_DICT_OBSERVATION_SPACE),
            FLATTENEND_DICT_SIZE
        )
    ]
    )
def test_dict_flatten_observation_v0(env, flattened_size):
    wrapped_env = flatten_observations_v0(env)
    obs, _, _, _ = wrapped_env.step(DISCRETE_VALUE)

    assert wrapped_env.observation_space.shape[0] == flattened_size
    assert obs.shape[0] == flattened_size
