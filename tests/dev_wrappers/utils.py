from typing import Optional

import gym
from gym import Space
from gym.core import ActType, ObsType
from gym.spaces import Box, Discrete


class TestingEnvironment(gym.Env):
    def __init__(
        self,
        observation_space: Space = Box(-10, 10, ()),
        action_space: Space = Discrete(5),
        reward_space: Space = Discrete(10),
        env_length: Optional[int] = None,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.env_length = env_length
        self.steps_left = env_length

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        if self.env_length is None:
            return (
                self.observation_space.sample(),
                self.reward_space.sample(),
                False,
                {},
            )
        else:
            self.steps_left -= 1
            return (
                self.observation_space.sample(),
                self.reward_space.sample(),
                self.steps_left == 0,
                {},
            )
