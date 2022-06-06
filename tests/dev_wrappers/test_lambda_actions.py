"""

"""
import pytest
import numpy as np

import gym
from gym.spaces import Dict, MultiBinary, MultiDiscrete, Tuple, Discrete, Box
from gym.wrappers import clip_actions_v0, lambda_action_v0, scale_actions_v0
from tests.dev_wrappers.utils import TestingEnv

ENVS = (
    gym.make("CartPole-v1", disable_env_checker=True),  # action_shape=Discrete(2)
    gym.make("MountainCarContinuous-v0", disable_env_checker=True),  # action_shape=Box(-1.0, 1.0, (1,), float32)
    gym.make('BipedalWalker-v3', disable_env_checker=True),  # action_shape=Box(-1.0, 1.0, (4,), float32)
    gym.vector.make("CartPole-v1", disable_env_checker=True),
    gym.vector.make("MountainCarContinuous-v0", disable_env_checker=True),
    gym.vector.make('BipedalWalker-v3', disable_env_checker=True),
    TestingEnv(action_space=MultiDiscrete([1, 2, 3])),
    TestingEnv(action_space=MultiBinary(5)),
    TestingEnv(action_space=MultiBinary([3, 3])),
    TestingEnv(action_space=Dict(left_arm=Discrete(4), right_arm=Box(0, 3, ()))),
    TestingEnv(action_space=Dict(body=Dict(left_arm=Discrete(1), right_arm=MultiBinary(3)),
                                 head=Box(0, 1, ()))),
    TestingEnv(action_space=Dict(hand=Tuple([Box(0, 1, ()), Discrete(1), Discrete(3)]),
                                 head=Box(0, 1, ()))),
    TestingEnv(action_space=Tuple([Box(0, 1, ()), Discrete(3)])),
    TestingEnv(action_space=Tuple([Tuple([Box(0, 1, ()), Discrete(3)]), Discrete(1)])),
    TestingEnv(action_space=Tuple([Dict(body=Box(0, 1, ())), Discrete(4)])),
)

SEED = 1


@pytest.mark.skip(reason="TODO")
@pytest.mark.parametrize("env", ENVS, ids=[str(env) for env in ENVS])
@pytest.mark.parametrize("args", [

])
def test_clip_actions_v0(env, args):
    wrapped_env = clip_actions_v0(env, args)

    wrapped_env.reset(seed=SEED)
    for _ in range(5):
        action = env.action_space.sample()
        wrapped_env.step(action)

@pytest.mark.skip(reason="TODO")
@pytest.mark.parametrize("env", ENVS)
@pytest.mark.parametrize("args", [

])
def test_scaled_actions_v0(env, args):
    wrapped_env = scale_actions_v0(env, args)

    wrapped_env.reset(seed=SEED)
    for _ in range(5):
        action = env.action_space.sample()
        wrapped_env.step(action)


@pytest.mark.parametrize(
    ("env", "fn", "action"),
    [
        (
            TestingEnv(action_space=Box(-1,1,(1,))),
            lambda action, _: action.astype(np.int32),
            np.float64(10),
        ),
    ]
)
def test_lambda_action_v0(env, fn, action):
    wrapped_env = lambda_action_v0(env, fn, None)
    _, _, _, info = wrapped_env.step(action)
    executed_action = info["action"]

    assert type(executed_action) == type(fn(action, None))


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
    ('env', 'args', "action"),
    [
        (
            TestingEnv(
                action_space=Dict(
                    left_arm=Discrete(4), 
                    right_arm=Box(0, 5, (1,))
                )
            ),
            {"right_arm": (0, 2)},
            {"right_arm": 10, "left_arm": 5}
        )
    ]
)
def test_clip_actions_v0_dict_testing_env(env, args, action):
    """Checks Dict action spaces clipping.

    Check wether dictionaries action spaces are
    correctly clipped.    
    """
    wrapped_env = clip_actions_v0(env, args)
    _, _, _, info = wrapped_env.step(action)
    executed_actions = info["action"]

    for k in executed_actions:
        if k in args:
            assert executed_actions[k] == args[k][1]
        else:
            assert executed_actions[k] == action[k]