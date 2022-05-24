from __future__ import annotations

from typing import Callable, Optional

import jumpy as jp

import gym


class lambda_reward_v0(gym.RewardWrapper):
    """TODO

    Example:
        TODO

    """
    def __init__(self, env: gym.Env, fn: Callable[[float], float]):
        super().__init__(env)

        self.fn = fn

    def reward(self, reward):
        return self.fn(reward)


class clip_rewards_v0(lambda_reward_v0):
    """TODO

    Example:
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
        elif not (min_reward < max_reward):
            raise Exception(
                f"Min reward ({min_reward}) must be less than max reward ({max_reward})"  # TODO update exception
            )
        else:
            super().__init__(
                env, lambda x: jp.clip(x, a_min=min_reward, a_max=max_reward)
            )


class normalize_rewards_v0(lambda_reward_v0):
    """TODO

    Example:
        TODO
    """
    def __init__(self):
        pass

