"""Test lambda reward wrapper."""
import numpy as np
import pytest

import gym
from gym.wrappers import clip_rewards_v0, lambda_reward_v0

ENV_ID = "CartPole-v1"
NUM_ENVS = 3
SEED = 1


@pytest.mark.parametrize(
    (
        "reward_fn",
        "expected_reward",
    ),
    [(lambda r: 2 * r + 1, 3)],
)
def test_lambda_reward(reward_fn, expected_reward):
    env = gym.make(ENV_ID)
    env = lambda_reward_v0(env, reward_fn)
    env.reset(seed=SEED)
    _, rew, _, _ = env.step(0)
    assert rew == expected_reward


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0, 0.5, 0.5)],
)
def test_clip_reward(lower_bound, upper_bound, expected_reward):
    env = gym.make(ENV_ID)
    env = clip_rewards_v0(env, lower_bound, upper_bound)
    env.reset(seed=SEED)
    _, rew, _, _ = env.step(0)

    assert rew == expected_reward


@pytest.mark.parametrize(("lower_bound", "upper_bound"), [(None, None), (1, -1)])
def test_clip_reward_incorrect_params(lower_bound, upper_bound):
    env = gym.make(ENV_ID)
    with pytest.raises(Exception):
        env = clip_rewards_v0(env, lower_bound, upper_bound)


@pytest.mark.parametrize(
    (
        "reward_fn",
        "expected_reward",
    ),
    [(lambda r: 2 * r + 1, 3)],
)
def test_lambda_reward_within_vector(reward_fn, expected_reward):
    ACTIONS = [0 for _ in range(NUM_ENVS)]

    env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    env = lambda_reward_v0(env, reward_fn)
    env.reset(seed=SEED)
    _, rew, _, _ = env.step(ACTIONS)
    assert np.alltrue(rew == expected_reward)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0, 0.5, 0.5)],
)
def test_clip_reward_within_vector(lower_bound, upper_bound, expected_reward):
    ACTIONS = [0 for _ in range(NUM_ENVS)]

    env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    env = clip_rewards_v0(env, lower_bound, upper_bound)
    env.reset(seed=SEED)
    _, rew, _, _ = env.step(ACTIONS)

    assert np.alltrue(rew == expected_reward)
