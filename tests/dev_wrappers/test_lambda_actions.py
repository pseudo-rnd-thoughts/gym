import pytest

import gym
from gym.spaces import Dict, MultiBinary, MultiDiscrete, Tuple
from gym.wrappers import clip_actions_v0, lambda_action_v0, scale_actions_v0
from tests.dev_wrappers.utils import TestingEnvironment

action_testing_environments = (
    gym.make("CartPole-v0"),
    gym.make(""),
    gym.make(""),
    gym.make(""),
    TestingEnvironment(action_space=MultiDiscrete()),
    TestingEnvironment(action_space=MultiBinary()),
    TestingEnvironment(action_space=Dict()),
    TestingEnvironment(action_space=Tuple()),
)


@pytest.mark.parametrize()
def test_lambda_action_v0(env, fn, args, expected_result):
    wrapped_env = lambda_action_v0(env, fn, args)


@pytest.mark.parametrize()
def test_clip_reward_v0(env, args):
    wrapped_env = clip_actions_v0(env)


@pytest.mark.parametrize()
def test_scaled_actions_v0(env):
    wrapped_env = scale_actions_v0(env)
