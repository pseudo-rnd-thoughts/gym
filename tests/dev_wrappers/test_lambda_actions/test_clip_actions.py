"""Test suite for clip_actions_v0."""
import numpy as np
import pytest

import gym
from gym.spaces import Dict
from gym.dev_wrappers.lambda_action import clip_actions_v0
from tests.dev_wrappers.test_lambda_actions.mock_data_actions import (
    NUM_ENVS,
    NEW_BOX_HIGH,
    NEW_BOX_LOW,
    NEW_NESTED_BOX_HIGH,
    NEW_NESTED_BOX_LOW,
    SEED,
    TESTING_DICT_ACTION_SPACE,
    TESTING_DICT_WITHIN_TUPLE_ACTION_SPACE,
    TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE,
    TESTING_DOUBLY_NESTED_TUPLE_ACTION_SPACE,
    TESTING_NESTED_DICT_ACTION_SPACE,
    TESTING_NESTED_TUPLE_ACTION_SPACE,
    TESTING_TUPLE_ACTION_SPACE,
    TESTING_TUPLE_WITHIN_DICT_ACTION_SPACE,
)
from tests.dev_wrappers.utils import TestingEnv


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
            (-0.5, 0.5),
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
        [
            "CarRacing-v1", # action space Box([-1.  0.  0.], 1.0, (3,), float32)
            (None, 0.5),
            np.array([0.5, 0.5, 0.5]),
            np.array([1, 1, 1])
        ]
    ),
)
def test_clip_actions_v0(env_name, args, action_unclipped_env, action_clipped_env):
    """Tests if actions out of bound are correctly clipped.

    Tests whether out of bound actions for the wrapped
    environments are correctly clipped.
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
    ("env_name", "args", "action_unclipped_env", "action_clipped_env"),
    (
        [
            "BipedalWalker-v3",
            (
                np.tile(np.array([-0.5, -0.5, -0.5, -0.5], dtype="float32"), (NUM_ENVS, 1)),
                np.tile(np.array([0.5, 0.5, 0.5, 0.5], dtype="float32"), (NUM_ENVS, 1)),
            ),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10, 10, 10, 10]),
        ],
    ),
)
def test_clip_actions_v0_vector_env(env_name, args, action_unclipped_env, action_clipped_env):
    """Tests clipped actions in vectorized environment.

    Tests whether out of bound actions 
    are correctly clipped in vectorized environment.
    """
    env = gym.vector.make(env_name, num_envs=NUM_ENVS)
    env.reset(seed=SEED)
    obs, _, _, _ = env.step([action_unclipped_env for _ in range(NUM_ENVS)])

    env = gym.vector.make(env_name, num_envs=NUM_ENVS)
    env.reset(seed=SEED)
    wrapped_env = clip_actions_v0(env, args)
    wrapped_obs, _, _, _ = wrapped_env.step([action_clipped_env for _ in range(NUM_ENVS)])

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
def test_clip_actions_v0_dict_action(env, args, action):
    """Checks `Dict` action spaces clipping.

    Check whether actions in `Dict` action spaces are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["box"] == NEW_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (  # one level of nested dict
            TestingEnv(action_space=TESTING_NESTED_DICT_ACTION_SPACE),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {"nested": (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)},
            },
            {
                "box": NEW_BOX_HIGH + 1,
                "discrete": 0,
                "nested": {"nested": NEW_NESTED_BOX_HIGH + 1},
            },
        ),
        (  # two levels of nested dict
            TestingEnv(action_space=TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE),
            {
                "box": (NEW_BOX_LOW, NEW_BOX_HIGH),
                "nested": {
                    "nested": {"nested": (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)}
                },
            },
            {
                "box": NEW_BOX_HIGH + 1,
                "nested": {"nested": {"nested": NEW_NESTED_BOX_HIGH + 1}},
            },
        ),
    ],
)
def test_clip_actions_v0_nested_dict_action(env, args, action):
    """Checks nested `Dict` action spaces clipping.

    Check whether actions in nested `Dict` action spaces are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    nested_action = executed_actions["nested"]
    while isinstance(nested_action, Dict):
        nested_action = nested_action["nested"]

    assert executed_actions["box"] == NEW_BOX_HIGH
    assert nested_action == NEW_NESTED_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_TUPLE_ACTION_SPACE),
            [None, (NEW_BOX_LOW, NEW_BOX_HIGH)],
            [0, NEW_BOX_HIGH + 1],
        )
    ],
)
def test_clip_actions_v0_tuple_action(env, args, action):
    """Checks `Tuple` action spaces clipping.

    Check whether actions in `Tuple` action spaces are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert np.alltrue(executed_actions == (0, NEW_BOX_HIGH))


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (  # one level of nesting
            TestingEnv(action_space=TESTING_NESTED_TUPLE_ACTION_SPACE),
            [
                (NEW_BOX_LOW, NEW_BOX_HIGH),
                [None, (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)],
            ],
            [NEW_BOX_HIGH + 1, [0, NEW_NESTED_BOX_HIGH + 1]],
        ),
        (  # two levels of nesting
            TestingEnv(action_space=TESTING_DOUBLY_NESTED_TUPLE_ACTION_SPACE),
            [
                (NEW_BOX_LOW, NEW_BOX_HIGH),
                [None, [None, (NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH)]],
            ],
            [NEW_BOX_HIGH + 1, [0, [0, (NEW_NESTED_BOX_HIGH + 1)]]],
        ),
    ],
)
def test_clip_actions_v0_nested_tuple_action(env, args, action):
    """Checks nested `Tuple` action spaces clipping.

    Check whether actions in nested `Tuple` action spaces are
    correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    nested_action = executed_actions[-1]
    while isinstance(nested_action, tuple):
        nested_action = nested_action[-1]

    assert executed_actions[0] == NEW_BOX_HIGH
    assert nested_action == NEW_NESTED_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_DICT_WITHIN_TUPLE_ACTION_SPACE),
            [None, {"dict": (NEW_BOX_LOW, NEW_BOX_HIGH)}],
            [0, {"dict": NEW_BOX_HIGH + 1}],
        )
    ],
)
def test_clip_actions_v0_dict_within_tuple(env, args, action):
    """Checks `Dict` within `Tuple` action spaces clipping.

    Check whether actions performed in `Dict` action space
    within an outer `Tuple` action space is correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions[1]["dict"] == NEW_BOX_HIGH


@pytest.mark.parametrize(
    ("env", "args", "action"),
    [
        (
            TestingEnv(action_space=TESTING_TUPLE_WITHIN_DICT_ACTION_SPACE),
            {"tuple": [(NEW_BOX_LOW, NEW_BOX_HIGH)]},
            {"discrete": 0, "tuple": [NEW_BOX_HIGH + 1]},
        )
    ],
)
def test_clip_actions_v0_tuple_within_dict(env, args, action):
    """Checks Tuple within Dict action spaces clipping.

    Check whether actions performed in `Tuple` action space
    within an outer `Dict` action space is correctly clipped.
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    assert executed_actions["tuple"][0] == NEW_BOX_HIGH
