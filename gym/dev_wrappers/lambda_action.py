from typing import Callable, Any

import jumpy as jp

import gym
from gym import Space
from gym.spaces import apply_function


class lambda_action_v0(gym.ActionWrapper):
    """A wrapper that provides a function to modify the action passed to :meth:`step`.

    Example to convert continuous actions to discrete:
        >>> import gym
        >>> import numpy as np
        >>> env = gym.make('CarRacing-v2')
        >>> env = lambda_action_v0(env, lambda action, _: np.astype(action, np.int32), None)
        >>>

    Composite action shape:
        TODO
    """

    def __init__(
        self, env: gym.Env, func: Callable, func_args: Any,
    ):
        super().__init__(env)

        self.func = func
        self.func_args = func_args

    def action(self, action):
        return apply_function(self.env.action_space, action, self.func, self.func_args)


class clip_actions_v0(lambda_action_v0):
    """A wrapper that clips actions passed to :meth:`step` with an upper and lower bound.

    Basic Example:
        >>> import gym
        >>> env = gym.make()
        >>> env = clip_actions_v0()

    Clip with only a lower or upper bound:
        >>> env = gym.make()
        >>> env = clip_actions_v0()

    Composite action space example:
        >>> env = gym.make()
        >>> env = clip_actions_v0()
    """

    def __init__(self, env: gym.Env, args, updated_action_space: Space):
        """

        Args:
            env: The environment to wrap
            args: TODO
            updated_action_space: TODO
        """
        super().__init__(env, lambda action, arg: jp.clip(action, *args), args)
        self.action_space = updated_action_space


class scale_actions_v0(lambda_action_v0):
    """TODO

    Example:
        TODO
    """

    def __init__(self, env: gym.Env, args, updated_action_space: Space):
        """TODO

        Args:
            env: The environment to wrap
            args: TODO
            updated_action_space: TODO
        """
        super().__init__(env, lambda action, arg: action * arg, args)
        self.action_space = updated_action_space
