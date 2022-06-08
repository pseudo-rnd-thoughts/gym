"""Test lambda_actions wrapper."""
import numpy as np
import pytest

import gym
from gym.dev_wrappers.lambda_action import scale_actions_v0
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple
from gym.wrappers import clip_actions_v0, lambda_action_v0  # scale_actions_v0
from tests.dev_wrappers.utils import TestingEnv, contains_space

ENVS = (
    gym.make("CartPole-v1", disable_env_checker=True),  # action_shape=Discrete(2)
    gym.make(
        "MountainCarContinuous-v0", disable_env_checker=True
    ),  # action_shape=Box(-1.0, 1.0, (1,), float32)
    gym.make(
        "BipedalWalker-v3", disable_env_checker=True
    ),  # action_shape=Box(-1.0, 1.0, (4,), float32)
    gym.vector.make("CartPole-v1", disable_env_checker=True),
    gym.vector.make("MountainCarContinuous-v0", disable_env_checker=True),
    gym.vector.make("BipedalWalker-v3", disable_env_checker=True),
    TestingEnv(action_space=MultiDiscrete([1, 2, 3])),
    TestingEnv(action_space=MultiBinary(5)),
    TestingEnv(action_space=MultiBinary([3, 3])),
    TestingEnv(action_space=Dict(left_arm=Discrete(4), right_arm=Box(0, 3, ()))),
    TestingEnv(
        action_space=Dict(
            body=Dict(left_arm=Discrete(1), right_arm=MultiBinary(3)),
            head=Box(0, 1, ()),
        )
    ),
    TestingEnv(
        action_space=Dict(
            hand=Tuple([Box(0, 1, ()), Discrete(1), Discrete(3)]), head=Box(0, 1, ())
        )
    ),
    TestingEnv(action_space=Tuple([Box(0, 1, ()), Discrete(3)])),
    TestingEnv(action_space=Tuple([Tuple([Box(0, 1, ()), Discrete(3)]), Discrete(1)])),
    TestingEnv(action_space=Tuple([Dict(body=Box(0, 1, ())), Discrete(4)])),
)


SEED = 1

DISCRETE_VALUE = 1

BOX_LOW, BOX_HIGH, BOX_DIM = -5, 5, 1
NEW_BOX_LOW, NEW_BOX_HIGH = 0, 2

NESTED_BOX_LOW, NESTED_BOX_HIGH = 0, 10
NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH = 0, 5


TESTING_BOX_ACTION_SPACE = Box(BOX_LOW, BOX_HIGH, (BOX_DIM,))

TESTING_DICT_ACTION_SPACE = Dict(
    discrete=Discrete(DISCRETE_VALUE),
    box=Box(BOX_LOW, BOX_HIGH, (BOX_DIM,)),
)

TESTING_NESTED_DICT_ACTION_SPACE = Dict(
    discrete=Discrete(DISCRETE_VALUE),
    box=Box(BOX_LOW, BOX_HIGH, (BOX_DIM,)),
    dict=Dict(nested=Box(NESTED_BOX_LOW, NESTED_BOX_HIGH, (BOX_DIM,))),
)


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
    wrapped_env = lambda_action_v0(env, fn, None)
    _, _, _, info = wrapped_env.step(action)
    executed_action = info["action"]

    assert isinstance(executed_action, type(fn(action, None)))


@pytest.mark.parametrize(
    ("env", "args"),
    (
        [
            gym.make("MountainCarContinuous-v0"),
            (np.array([-0.5], dtype="float32"), np.array([0.5], dtype="float32")),
        ],
        [
            gym.make("BipedalWalker-v3"),
            (
                -0.5,
                0.5,
            ),
        ],
        [
            gym.make("BipedalWalker-v3"),
            (
                np.array([-0.5, -1, -1, -1], dtype="float32"),
                np.array([0.5, 0.5, 1, 1], dtype="float32"),
            ),
        ],
    ),
)
def test_clip_actions_v0_wrapped_action_space(env, args):
    """Tests if the wrapped action space is correctly clipped.

    This tests assert that the action space of the
    wrapped environment is clipped correctly according
    the args parameters.
    """
    action_space_before_wrapping = env.action_space

    wrapped_env = clip_actions_v0(env, args)

    assert np.equal(wrapped_env.action_space.low, args[0]).all()
    assert np.equal(wrapped_env.action_space.high, args[1]).all()
    assert action_space_before_wrapping == wrapped_env.env.action_space


@pytest.mark.parametrize(
    ("env_name", "args", "action_unclipped_env", "action_clipped_env"),
    (
        [
            "MountainCarContinuous-v0",
            (np.array([-0.5], dtype="float32"), np.array([0.5], dtype="float32")),
            np.array([0.5]),
            np.array([1]),
        ],
        [
            "BipedalWalker-v3",
            (
                -0.5,
                0.5,
            ),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10, 10, 10, 10]),
        ],
        [
            "BipedalWalker-v3",
            (
                np.array([-0.5, -1, -1, -1], dtype="float32"),
                np.array([0.5, 0.5, 1, 1], dtype="float32"),
            ),
            np.array([0.5, 0.5, 1, 1]),
            np.array([10, 10, 10, 10]),
        ],
    ),
)
def test_clip_actions_v0(env_name, args, action_unclipped_env, action_clipped_env):
    """Tests if actions out of bound are correctly clipped.

    This tests check whether out of bound actions for the wrapped
    environment are correctly clipped.
    """
    env = gym.make(env_name)
    env.reset(seed=SEED)
    obs, _, _, _ = env.step(action_unclipped_env)

    env = gym.make(env_name)
    env.reset(seed=SEED)
    wrapped_env = clip_actions_v0(env, args)
    wrapped_obs, _, _, _ = wrapped_env.step(action_clipped_env)

    assert np.alltrue(obs == wrapped_obs)


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_DICT_ACTION_SPACE),
            {"box": (NEW_BOX_LOW, NEW_BOX_HIGH)},
            {"box": NEW_BOX_HIGH + 1},
        )
    ],
)
def test_clip_actions_v0_dict_testing_env(env, args, action):
    """Checks Dict action spaces clipping.

    Check whether dictionaries action spaces are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["box"] == NEW_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_NESTED_DICT_ACTION_SPACE),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "dict": {"nested": (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)},
            },
            {
                "box": NEW_BOX_HIGH + 1,
                "discrete": 0,
                "dict": {"nested": NEW_NESTED_BOX_HIGH + 1},
            },
        )
    ],
)
def test_clip_actions_v0_nested_dict(env, args, action):
    """Checks Nested Dict action spaces clipping.

    Check whether nested dictionaries action spaces are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["box"] == NEW_BOX_HIGH
    assert executed_actions["dict"]["nested"] == NEW_NESTED_BOX_HIGH


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
                "dict": {"nested": (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)},
            },
            {
                "box": NEW_BOX_HIGH,
                "discrete": 0,
                "dict": {"nested": NEW_NESTED_BOX_HIGH},
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
    assert executed_actions["dict"]["nested"] == NESTED_BOX_HIGH
