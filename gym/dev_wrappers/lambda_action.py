"""Lambda action wrappers that uses jumpy for compatibility with jax (i.e. brax) and numpy environments."""

from typing import Any, Callable
from typing import Tuple as TypingTuple

import jumpy as jp

import gym
from gym import Space
from gym.dev_wrappers import FuncArgType
from gym.dev_wrappers.utils.commons import extend_args
from gym.dev_wrappers.utils.transform_space_bounds import transform_space_bounds
from gym.spaces import Box, Dict, Tuple, apply_function


class lambda_action_v0(gym.ActionWrapper):
    """A wrapper that provides a function to modify the action passed to :meth:`step`.

    Example to convert continuous actions to discrete:
        >>> import gym
        >>> from gym.spaces import Dict
        >>> import numpy as np
        >>> env = gym.make('CarRacingDiscrete-v1')
        >>> env = lambda_action_v0(env, lambda action, _: action.astype(np.int32), None)
        >>> env.action_space
        Discrete(5)
        >>> _ = env.reset()
        >>> obs, rew, done, info = env.step(np.float64(1.2))

    Composite action shape:
        >>> env = ExampleEnv(action_space=Dict(left_arm=Discrete(4), right_arm=Box(0.0, 5.0, (1,)))
        >>> env = lambda_action_v0(
        ...     env,
        ...     lambda action, _: action + 10,
        ...     {"right_arm": True},
        ...     None
        ... )
        >>> env.action_space
        Dict(left_arm: Discrete(4), right_arm: Box(0.0, 5.0, (1,), float32))
        >>> _ = env.reset()
        >>> obs, rew, done, info = env.step({"left_arm": 1, "right_arm": 1})
        >>> info["action"] # the executed action whitin the environment
        {'action': OrderedDict([('left_arm', 1), ('right_arm', 11)])})
    """

    def __init__(
        self,
        env: gym.Env,
        func: Callable,
        args: FuncArgType[Any],
        action_space: Space = None,
    ):
        """Initialize lambda_action."""
        super().__init__(env)

        self.func = func
        self.func_args = args
        if action_space is None:
            self.action_space = env.action_space
        else:
            self.action_space = action_space

    def action(self, action):
        """Apply function to action."""
        return apply_function(self.action_space, action, self.func, self.func_args)

    def _transform_space(self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
        """Process the space and apply the transformation."""
        return transform_space_bounds(env.action_space, args, transform_space_bounds)


class clip_actions_v0(lambda_action_v0):
    """A wrapper that clips actions passed to :meth:`step` with an upper and lower bound.

    Basic Example:
        >>> import gym
        >>> env = gym.make("BipedalWalker-v3")
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env = clip_actions_v0(env, (-0.5, 0.5))
        >>> env.action_space
        Box(-0.5, 0.5, (4,), float32)

    Clip with only a lower or upper bound:
        >>> env = gym.make(TODO)
        >>> env.action_space
        TODO
        >>> env = clip_actions_v0(env, TODO)
        >>> env.action_space
        TODO

    Composite action space example:
        >>> env = ExampleEnv()
        >>> env.actions_space
        Dict(body: Dict(head: Box(0.0, 10.0, (1,), float32)), left_arm: Discrete(4), right_arm: Box(0.0, 5.0, (1,), float32))
        >>> args = {"right_arm": (0, 2), "body": {"head": (0, 3)}}
        >>> env = clip_actions_v0(env, args)
        >>> env.action_space
        Dict(body: Dict(head: Box(0.0, 3.0, (1,), float32)), left_arm: Discrete(4), right_arm: Box(0.0, 2.0, (1,), float32))
    """

    def __init__(self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
        """Constructor for the clip action wrapper.

        Args:
            env: The environment to wrap
            args: The arguments for clipping the action space
        """
        action_space = self._transform_space(env, args)

        super().__init__(
            env, lambda action, args: jp.clip(action, *args), args, action_space
        )


class scale_actions_v0(lambda_action_v0):
    """A wrapper that scales actions passed to :meth:`step` with a scale factor.

    Basic Example:
        >>> import gym
        >>> env = gym.make('BipedalWalker-v3')
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
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
        """Constructor for the scale action wrapper.

        Args:
            env: The environment to wrap
            args: The arguments for scaling the actions
        """
        action_space = self._transform_space(env, args)

        if isinstance(env.action_space, Box):
            args = (*args, env.action_space.low, env.action_space.high)

        elif isinstance(env.action_space, Dict):
            extended_args = {}
            for arg in args:
                extend_args(env.action_space, extended_args, args, arg)
            args = extended_args

        elif isinstance(env.action_space, Tuple):
            # TODO
            ...

        def func(action, args):
            new_low, new_high = args[0], args[1]
            old_low, old_high = args[2], args[3]

            return jp.clip(
                old_low
                + (old_high - old_low) * ((action - new_low) / (new_high - new_low)),
                old_low,
                old_high,
            )

        super().__init__(env, func, args, action_space)
