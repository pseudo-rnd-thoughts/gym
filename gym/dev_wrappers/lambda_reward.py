from typing import Callable, Optional, Union

import jumpy as jp
import numpy as np

import gym


class lambda_reward_v0(gym.RewardWrapper):
    """A reward wrapper that allows a custom function to modify the step reward

    Example:
        >>> import gym
        >>> env = gym.make("CartPole-v1")
        >>> env = lambda_reward_v0(env, lambda r: 2 * r + 1)
        >>> env.reset()
        >>> env.step(0)
        TODO
    """
    def __init__(self, env: gym.Env, fn: Callable[[Union[float, int, np.ndarray]], Union[float, int, np.ndarray]]):
        super().__init__(env)

        self.fn = fn

    def reward(self, reward):
        return self.fn(reward)


class clip_rewards_v0(lambda_reward_v0):
    """A wrapper that clips the rewards for an environment between an upper and lower bound.

    Example with an upper and lower bound:
        >>> import gym
        >>> env = gym.make("CartPole-v1")
        >>> env = clip_rewards_v0(env, -1, 5)
        >>> env.reset()
        >>> env.step(1)
        TODO
    """
    def __init__(
        self,
        env: gym.Env,
        min_reward: Optional[float | jp.ndarray] = None,
        max_reward: Optional[float | jp.ndarray] = None,
    ):
        if min_reward is None and max_reward is None:
            raise Exception("Both `min_reward` and `max_reward` cannot be None")  # TODO update exception
        elif max_reward < min_reward:
            raise Exception(
                f"Min reward ({min_reward}) must be less than max reward ({max_reward})"  # TODO update exception
            )
        else:
            super().__init__(
                env, lambda x: jp.clip(x, a_min=min_reward, a_max=max_reward)
            )
