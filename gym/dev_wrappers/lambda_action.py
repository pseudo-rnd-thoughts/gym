from typing import Callable, Any, Tuple as TypingTuple

import jumpy as jp

import gym
from gym import Space
from gym.dev_wrappers import FuncArgType
from gym.spaces import apply_function


class lambda_action_v0(gym.ActionWrapper):
    """A wrapper that provides a function to modify the action passed to :meth:`step`.

    Example to convert continuous actions to discrete:
        >>> import gym
        >>> from gym.spaces import Dict
        >>> import numpy as np
        >>> env = gym.make('CarRacing-v2')
        >>> env = lambda_action_v0(env, lambda action, _: np.astype(action, np.int32), None)
        >>> env.action_space
        TODO
        >>> env.reset()
        TODO
        >>> env.step()[0]
        TODO

    Composite action shape:
        >>> env = ExampleEnv(action_space=Dict(TODO))
        >>> env = lambda_action_v0(env, TODO, TODO, TODO)
        >>> env.action_space
        TODO
        >>> env.reset()
        TODO
        >>> env.step()[0]
        TODO
    """

    def __init__(
        self, env: gym.Env, func: Callable, args: FuncArgType[Any], action_space: Space = None
    ):
        super().__init__(env)

        self.func = func
        self.func_args = args
        if action_space is None:
            self.action_space = env.action_space
        else:
            self.action_space = action_space

    def action(self, action):
        return apply_function(self.env.action_space, action, self.func, self.func_args)


class clip_actions_v0(lambda_action_v0):
    """A wrapper that clips actions passed to :meth:`step` with an upper and lower bound.

    Basic Example:
        >>> import gym
        >>> env = gym.make("TODO")
        >>> env.action_space
        TODO
        >>> env = clip_actions_v0(env, TODO)
        >>> env.action_space
        TODO

    Clip with only a lower or upper bound:
        >>> env = gym.make(TODO)
        >>> env.action_space
        TODO
        >>> env = clip_actions_v0(env, TODO)
        >>> env.action_space
        TODO

    Composite action space example:
        >>> env = ExampleEnv()
        >>> env = clip_actions_v0(env, TODO)
        >>> env.action_space
        TODO
    """

    def __init__(self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
        """Constructor for the clip action wrapper

        Args:
            env: The environment to wrap
            args: The arguments for clipping the action space
        """
        action_space = None  # TODO update action space
        super().__init__(env, lambda action, arg: jp.clip(action, *args), args, action_space)


class scale_actions_v0(lambda_action_v0):
    """A wrapper that scales actions passed to :meth:`step` with a scale factor

    Basic Example:
        >>> import gym
        >>> env = gym.make(TODO)
        >>> env.action_space
        TODO
        >>> env = scale_actions_v0(env, TODO, TODO)
        >>> env.action_space
        TODO

    Composite action space example:
        >>> env = ExampleEnv()
        >>> env = scale_actions_v0(env, TODO, TODO)
        >>> env.action_space
        TODO
    """

    def __init__(self, env: gym.Env, args: FuncArgType[float]):
        """Constructor for the scale action wrapper

        Args:
            env: The environment to wrap
            args: The arguments for scaling the actions
        """
        action_space = None  # TODO, add action space
        super().__init__(env, lambda action, arg: action * arg, args, action_space)
