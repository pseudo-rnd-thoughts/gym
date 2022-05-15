from typing import Callable, Optional

import jumpy as jp

import gym
from gym.core import ActType, ObsType


class lambda_action_v0(gym.ActionWrapper):
    def __init__(
        self, env: gym.Env[ObsType, ActType], fn: Callable[[ActType], ActType], fn_args
    ):
        super().__init__(env)

        self.fn = fn

    def action(self, action: ActType):
        return self.fn(action)


class clip_actions_v0(lambda_action_v0):
    pass


class scale_actions_v0(lambda_action_v0):
    pass
