"""

"""
import pytest

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


@pytest.mark.parametrize('env', ENVS)
@pytest.mark.parametrize('fn', [lambda x, arg: 2 * x + 1])
def test_lambda_action_v0(env, fn, args, expected_result):
    wrapped_env = lambda_action_v0(env, fn, args)


@pytest.mark.parametrize("env", ENVS, ids=[str(env) for env in ENVS])
@pytest.mark.parametrize("args", [

])
def test_clip_actions_v0(env, args):
    wrapped_env = clip_actions_v0(env, args)

    wrapped_env.reset(seed=SEED)
    for _ in range(5):
        action = env.action_space.sample()
        wrapped_env.step(action)


@pytest.mark.parametrize("env", ENVS)
@pytest.mark.parametrize("args", [

])
def test_scaled_actions_v0(env, args):
    wrapped_env = scale_actions_v0(env, args)

    wrapped_env.reset(seed=SEED)
    for _ in range(5):
        action = env.action_space.sample()
        wrapped_env.step(action)
