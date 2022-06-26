import pytest

import gym
from gym.wrappers import resize_observations_v0
from tests.dev_wrappers.test_lambda_observations.mock_data_observation import (
    SEED,
    TESTING_DICT_OBSERVATION_SPACE,
    TESTING_TUPLE_OBSERVATION_SPACE,
)
from tests.dev_wrappers.utils import TestingEnv


@pytest.mark.parametrize(
    (
        "env",
        "args",
    ),
    [
        (
            gym.make("CarRacingDiscrete-v1"),  # Box(0, 255, (96, 96, 3), uint8)
            (32, 32, 3),
        )
    ],
)
def test_resize_observations_box_v0(env, args):
    """Test correct resizing of box observations."""
    wrapped_env = resize_observations_v0(env, args)
    wrapped_env.reset(seed=SEED)

    assert wrapped_env.observation_space.shape == args

    action = wrapped_env.action_space.sample()
    obs, *res = wrapped_env.step(action)
    assert obs.shape == args


@pytest.mark.parametrize(
    (
        "env",
        "args",
    ),
    [
        (
            TestingEnv(observation_space=TESTING_TUPLE_OBSERVATION_SPACE),
            [(5, 5), (2, 2)],
        ),
        (TestingEnv(observation_space=TESTING_TUPLE_OBSERVATION_SPACE), [(5, 5), None]),
        (TestingEnv(observation_space=TESTING_TUPLE_OBSERVATION_SPACE), [None, (5, 5)]),
    ],
)
def test_resize_observations_tuple_v0(env, args):
    """Test correct resizing of `Tuple` observations."""
    wrapped_env = resize_observations_v0(env, args)
    wrapped_env.reset(seed=SEED)

    action = wrapped_env.action_space.sample()
    obs, *res = wrapped_env.step(action)

    for i, arg in enumerate(args):
        if not arg:
            assert (
                wrapped_env.observation_space[i].shape == env.observation_space[i].shape
            )
            assert obs[i].shape == env.observation_space[i].shape
        else:
            assert wrapped_env.observation_space[i].shape == arg
            assert obs[i].shape == arg


@pytest.mark.parametrize(
    (
        "env",
        "args",
    ),
    [
        (
            TestingEnv(observation_space=TESTING_DICT_OBSERVATION_SPACE),
            {"key_1": (5, 5)},
        ),
        (
            TestingEnv(observation_space=TESTING_DICT_OBSERVATION_SPACE),
            {"key_1": (5, 5), "key_2": (2, 2)},
        ),
    ],
)
def test_resize_observations_dict_v0(env, args):
    """Test correct resizing of `Dict` observations."""
    wrapped_env = resize_observations_v0(env, args)
    wrapped_env.reset(seed=SEED)

    action = wrapped_env.action_space.sample()
    obs, *res = wrapped_env.step(action)

    for k in obs:
        if k in args:
            assert wrapped_env.observation_space[k].shape == args[k]
            assert obs[k].shape == args[k]
        else:
            assert (
                wrapped_env.observation_space[k].shape == env.observation_space[k].shape
            )
            assert obs[k].shape == env.observation_space[k].shape
