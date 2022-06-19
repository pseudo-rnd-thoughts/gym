"""Test suite for scale_actions_v0."""
import numpy as np
import pytest

import gym
from gym.dev_wrappers.lambda_action import scale_actions_v0
from tests.dev_wrappers.test_lambda_actions import (
    BOX_HIGH,
    NESTED_BOX_HIGH,
    NEW_BOX_HIGH,
    NEW_BOX_LOW,
    NEW_NESTED_BOX_HIGH,
    NEW_NESTED_BOX_LOW,
    SEED,
    TESTING_NESTED_DICT_ACTION_SPACE,
)
from tests.dev_wrappers.utils import TestingEnv


def test_scale_actions_v0_box():
    """Test action rescaling.

    Scale action wrapper allow to rescale action
    to a new range.
    Supposed the old action space is of type
    `Box(-1, 1, (1,))` and we rescale to
    `Box(-0.5, 0.5, (1,))`, an action on the wrapped
    environment with value `0.5` will be seen by the
    environment as `1.0`.  TODO: maybe explain this better.
    """

    ENV_ID = "BipedalWalker-v3"
    SCALE_LOW, SCALE_HIGH = -0.5, 0.5
    ARGS = (SCALE_LOW, SCALE_HIGH)

    ACTION = np.array([1, 1, 1, 1])
    RESCALED_ACTION = np.array([SCALE_HIGH, SCALE_HIGH, SCALE_HIGH, SCALE_HIGH])

    env = gym.make(ENV_ID)
    env.reset(seed=SEED)
    obs, _, _, _ = env.step(ACTION)

    env = gym.make(ENV_ID)
    wrapped_env = scale_actions_v0(env, ARGS)
    wrapped_env.reset(seed=SEED)
    obs_scaled, _, _, _ = wrapped_env.step(RESCALED_ACTION)

    assert np.alltrue(obs == obs_scaled)


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_NESTED_DICT_ACTION_SPACE),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {"nested": (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)},
            },
            {
                "box": NEW_BOX_HIGH,
                "discrete": 0,
                "nested": {"nested": NEW_NESTED_BOX_HIGH},
            },
        )
    ],
)
def test_scale_actions_v0_nested_dict(env, args, action):
    """Test action rescaling for nested dict action spaces."""

    wrapped_env = scale_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["box"] == BOX_HIGH
    assert executed_actions["nested"]["nested"] == NESTED_BOX_HIGH
