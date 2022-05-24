from typing import Optional, Union

import gym
from gym import Space
from gym.core import ActType, ObsType
from gym.spaces import Box, Discrete, Dict, Tuple


class TestingEnv(gym.Env):
    """A testing environment for wrappers where a custom action or observation can be passed to the environment.

    The action and observation spaces provided are used to sample new observations or actions to test with the environment
    """
    def __init__(
        self,
        observation_space: Space = Box(-10, 10, ()),
        action_space: Space = Discrete(5),
        reward_space: Space = Discrete(10),
        env_length: Optional[int] = None,
    ):
        """Constructor of the testing environment

        Args:
            observation_space: The environment observation shape
            action_space: The environment action space
            reward_space: The reward space for the environment to sample from
            env_length: The environment length used to know if the environment has timed out
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.reward_space = reward_space
        self.env_length = env_length
        self.steps_left = env_length

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None) -> Union[ObsType, tuple[ObsType, dict]]:
        """TODO"""
        self.steps_left = self.env_length
        if return_info:
            return self.observation_space.sample(), {}
        else:
            return self.observation_space.sample()

    def step(self, action: ActType) -> tuple[ObsType, float, bool, dict]:
        """"TODO"""
        if self.env_length is not None:
            self.steps_left -= 1

        return (
            self.observation_space.sample(),
            self.reward_space.sample(),
            self.env_length is not None and self.steps_left == 0,
            {}
        )

    def id(self):
        pass


def contains_space(space: Space, contain_type: type) -> bool:
    """Checks if a space is or contains a space type"""
    if isinstance(space, contain_type):
        return True
    elif isinstance(space, Dict):
        return any(contains_space(subspace, contain_type) for subspace in space.values())
    elif isinstance(space, Tuple):
        return any(contains_space(subspace, contain_type) for subspace in space.spaces)
    else:
        return False
