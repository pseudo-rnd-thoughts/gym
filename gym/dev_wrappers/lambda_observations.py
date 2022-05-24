from typing import Any, Callable, Optional

import jumpy as jp
import tinyscaler

import gym
from gym import Space
from gym.core import ObsType
from gym.spaces.utils import apply_function, flatten


class lambda_observations_v0(gym.ObservationWrapper):
    """Lambda observation wrapper where a function is provided that is applied to the observation.

    Example:
        TODO

    """

    def __init__(
        self,
        env: gym.Env,
        func: Callable[[ObsType, Any], ObsType],
        args: Any,
        updated_observation_shape: Optional[Space] = None,
    ):
        """Constructor for the lambda observation wrapper

        Args:
            env: The environment to wrap
            func: A function that takes
            args: The arguments that the function takes
            updated_observation_shape: The updated observation shape
        """
        super().__init__(env)
        self.func = func
        self.args = args

        if updated_observation_shape:
            self.observation_space = updated_observation_shape

    def observation(self, observation: ObsType):
        return apply_function(self.observation_space, observation, self.func, self.args)


class filter_observations_v0(lambda_observations_v0):
    """Filter Dict or Tuple observation space by the keys or indexes respectively.

    Example with Dict observation:
        >>> import gym
        >>> from gym.spaces import Tuple, Dict, Discrete, Box
        >>> env = ExampleEnv(observation_shape=Dict(obs=Box(), time=Discrete()))
        >>> env = filter_observations_v0(env, ['obs'])
        >>> env.reset()

        >>> env = ExampleEnv(observation_shape=Dict())
        >>> env = filter_observations_v0(env, {'obs': True, 'time': False})
        >>> env.reset()

    Example with Tuple observation:
        >>> env = ExampleEnv(observation_shape=Tuple())
        >>> env = filter_observations_v0(env, [0, 3])
        >>> env.reset()

        >>> env = ExampleEnv(observation_shape=Tuple())
        >>> env = filter_observations_v0(env, [True, False, False, True])
        >>> env.reset()

    Example with three-order observation space:
        >>> env = ExampleEnv(observation_shape=Tuple([Tuple([Discrete(), Box()]), Dict()]))
        >>> env = filter_observations_v0(env, [[True, False], {}])
        >>> env.reset()

        >>> env = ExampleEnv(observation_shape=Dict(x=Tuple([Discrete(), Box()]), y=Dict()))
        >>> env = filter_observations_v0(env, {'x': [True, False], 'y': {}}, None)
        >>> env.reset()

    """

    def __init__(
        self, env, func_args, updated_observation_shape: Space
    ):
        """

        Args:
            env: The environment to wrap
            func_args:
            updated_observation_shape:
        """
        super().__init__(
            env, lambda obs, arg: obs, func_args, updated_observation_shape
        )


class flatten_observations_v0(lambda_observations_v0):
    """A wrapper that flattens observations returned by :meth:`step` and :meth:`reset`.

    """

    def __init__(
        self, env, func_args, updated_observation_shape: Space
    ):
        """

        Args:
            env: The environment to wrap
            func_args:
            updated_observation_shape:
        """
        super().__init__(
            env,
            lambda obs, arg: obs if arg is False else flatten(space, obs),
            func_args,
            updated_observation_shape,
        )


class grayscale_observations_v0(lambda_observations_v0):
    """A wrapper that converts color observations to grayscale that are returned by :meth:`step` and :meth:`reset`.

    """

    def __init__(
        self, env, func_args, updated_observation_shape: Space
    ):
        """

        Args:
            env: The environment to wrap
            func_args:
            updated_observation_shape:
        """
        super().__init__(
            env,
            lambda obs, arg: obs if arg is None else grayscale(obs),
            func_args,
            updated_observation_shape,
        )


class normalize_observations_v0(lambda_observations_v0):
    """A wrapper that normalizes an observation returned by :meth:`step` and :meth:`reset` with a mean and standard deviation.

    """

    def __init__(
        self, env, func_args, updated_observation_shape: Optional[Space] = None
    ):
        """

        Args:
            env: The environment to wrap
            func_args:
            updated_observation_shape:
        """
        super().__init__(env, TODO, func_args, updated_observation_shape)


class resize_observations_v0(lambda_observations_v0):
    """A wrapper that resizes an observation returned by :meth:`step` and :meth:`reset` to a new shape.

    Basic Example:
        TODO

    Composite Example:
        TODO
    """

    def __init__(self, env, func_args, updated_observation_shape: Space):
        """

        Args:
            env: The environment to wrap
            func_args:
            updated_observation_shape:
        """
        super().__init__(
            env,
            lambda obs, arg: obs if arg is None else tinyscaler.scale(obs, *arg),
            func_args,
            updated_observation_shape,
        )


class reshape_observation_v0(lambda_observations_v0):
    """TODO

    """

    def __init__(self, env, func_args, updated_observation_shape: Space):
        """

        Args:
            env: The environment to wrap
            func_args:
            updated_observation_shape:
        """
        super().__init__(
            env,
            lambda obs, arg: obs if arg is None else jp.reshape(obs, arg),
            func_args,
            updated_observation_shape,
        )


class observation_dtype_v0(lambda_observations_v0):
    """TODO

    """

    def __init__(
        self, env, func_args, updated_observation_shape: Space
    ):
        """

        Args:
            env: The environment to wrap
            func_args:
            updated_observation_shape:
        """
        super().__init__(env, lambda obs, arg: jp.astype(obs, arg), func_args, updated_observation_shape)
