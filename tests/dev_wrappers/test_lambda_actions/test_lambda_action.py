"""Test suite for lambda_action_v0."""
import numpy as np
import pytest

from gym.dev_wrappers.lambda_action import lambda_action_v0
from tests.dev_wrappers.test_lambda_actions.mock_data_actions import TESTING_BOX_ACTION_SPACE
from tests.dev_wrappers.utils import TestingEnv


@pytest.mark.parametrize(
    ("env", "fn", "action"),
    [
        (
            TestingEnv(action_space=TESTING_BOX_ACTION_SPACE),
            lambda action, _: action.astype(np.int32),
            np.float64(10),
        ),
    ],
)
def test_lambda_action_v0(env, fn, action):
    """Checks if lambda function is correctly applied to action."""
    wrapped_env = lambda_action_v0(env, fn, None)
    _, _, _, info = wrapped_env.step(action)
    executed_action = info["action"]

    assert isinstance(executed_action, type(fn(action, None)))
