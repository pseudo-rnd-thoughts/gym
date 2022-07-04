"""Lambda observation wrappers that uses jumpy for compatibility with jax (i.e. brax) and numpy environments."""
from typing import Any, Callable, Optional
from typing import Tuple as TypingTuple
from typing import Union

import jumpy as jp
import tinyscaler

import gym
from gym import spaces
from gym.core import ObsType
from gym.dev_wrappers import ArgType, FuncArgType
from gym.dev_wrappers.utils.filter_space import filter_space
from gym.dev_wrappers.utils.grayscale_space import grayscale_space
from gym.dev_wrappers.utils.reshape_space import reshape_space
from gym.dev_wrappers.utils.resize_spaces import resize_space
from gym.spaces.utils import apply_function, flatten, flatten_space


class lambda_observations_v0(gym.ObservationWrapper):
    """Lambda observation wrapper where a function is provided that is applied to the observation.

    Example:
        >>> import gym
        >>> from gym.spaces import Dict, Discrete
        >>> env = gym.make("CartPole-v1")
        >>> env.observation_space
        Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32)
        >>> env = lambda_observations_v0(
        ...     env,
        ...     lambda obs, arg: {"obs": obs, "time": 1},
        ...     None,
        ...     Dict(obs=env.action_space, time=Discrete(1))
        ... )
        >>> env.observation_space
        Dict(obs: Box([-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38], [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38], (4,), float32), time: Discrete(1))

    Composite observation space:
        >>> env = ExampleEnv(
        ...     observation_space=Dict(
        ...         left_arm=Box(-5, 5, (1,)),
        ...         right_arm=Box(-5, 5, (1,))
        ...     )
        ... )
        >>> env = lambda_observations_v0(
        ...     env,
        ...     lambda obs, arg: obs * arg,
        ...     {"left_arm": 0, "right_arm": float('inf')},
        ...     env.observation_space
        ... )
        >>> obs, _, _, _ = env.step({"left_arm": 1, "right_arm": 1}))
        >>> obs
        OrderedDict([('left_arm', array([0.], dtype=float32)), ('right_arm', array([-inf], dtype=float32))])
    """

    def __init__(
        self,
        env: gym.Env,
        func: Callable[[ObsType, ArgType], ObsType],
        args: FuncArgType[Any],
        observation_space: Optional[spaces.Space] = None,
    ):
        """Constructor for the lambda observation wrapper.

        Args:
            env: The environment to wrap
            func: A function that takes
            args: The arguments that the function takes
            observation_space: The updated observation space
        """
        super().__init__(env)
        self.func = func
        self.args = args
        if observation_space is None:
            self.observation_space = env.observation_space
        else:
            self.observation_space = observation_space

    def observation(self, observation: ObsType):
        """Apply function to the observation."""
        return apply_function(self.observation_space, observation, self.func, self.args)

    def _reshape_space(self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
        """Process the space and apply the transformation."""
        return reshape_space(env.observation_space, args, reshape_space)


class filter_observations_v0(lambda_observations_v0):
    """Filter Dict or Tuple observation space by the keys or indexes respectively.

    Example with Dict observation:
        >>> import gym
        >>> from gym.spaces import Tuple, Dict, Discrete, Box
        >>> env = ExampleEnv(observation_space=Dict(obs=Box(-1, 1, ()), time=Discrete(3)))
        >>> env = filter_observations_v0(env, ['obs'])
        >>> env.observation_space
        Dict(obs: Box(-1.0, 1.0, (), float32))

        >>> env = ExampleEnv(observation_space=Dict(obs=Box(-1, 1, ()), time=Discrete(3)))
        >>> env = filter_observations_v0(env, {'obs': True, 'time': False})
        >>> env.observation_space
        Dict(obs: Box(-1.0, 1.0, (), float32))

    Example with Tuple observation:
        >>> env = ExampleEnv(observation_space=Tuple([Box(-1, 1, ()), Box(-2, 2, ()), Discrete(3)]))
        >>> env = filter_observations_v0(env, [0, 2])
        >>> env.observation_space
        TODO

        >>> env = ExampleEnv(observation_space=Tuple([Box(-1, 1, ()), Box(-1, 1, ()), Box(-1, 1, ()), Box(-1, 1, ())]))
        >>> env = filter_observations_v0(env, [True, False, False, True])
        >>> env.observation_space
        TODO

    Example with three-order observation space:
        >>> env = ExampleEnv(observation_space=Tuple([Tuple([Discrete(3), Box(-1, 1, ())]),
        ...                                           Dict(obs=Box(-1, 1, ()), time=Discrete(3))]))
        >>> env = filter_observations_v0(env, [[True, False], ["obs"]])
        >>> env.observation_space
        TODO

        >>> env = ExampleEnv(observation_space=Tuple([Tuple([Discrete(3), Box(-1, 1, ())]),
        ...                                           Dict(obs=Box(-1, 1, ()), time=Discrete(3))]))
        >>> env = filter_observations_v0(env, [[True, False], {"obs": True, "time": False}])
        >>> env.observation_space
        TODO


        >>> env = ExampleEnv(observation_space=Dict(x=Tuple([Discrete(), Box()]), y=Dict()))
        >>> env = filter_observations_v0(env, {'x': [True, False], 'y': {}})
        >>> env.observation_space
        TODO
    """

    def __init__(self, env: gym.Env, args: FuncArgType[Union[str, int, bool]]):
        """Constructor for the filter observation wrapper.

        Args:
            env: The environment to wrap
            args: The filter arguments
        """
        # TODO: _filter_space is actually filtering space AND processing args
        # might need refactor
        observation_space, args = self._filter_space(env.observation_space, args)

        super().__init__(env, lambda obs, arg: obs, args, observation_space)

    def _filter_space(
        self, space: spaces.Space, args: FuncArgType[Union[str, int, bool]]
    ):
        """Filter space with the provided args."""
        return filter_space(space, args, filter_space)


class flatten_observations_v0(lambda_observations_v0):
    """A wrapper that flattens observations returned by :meth:`step` and :meth:`reset`.

    Basic Example, fully flattens the environment observation:
        >>> import gym
        >>> from gym.spaces import Dict, Box
        >>> env = gym.make("CarRacingDiscrete-v1")
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = flatten_observations_v0(env)
        >>> env.observation_space
        Box(0, 255, (27648,), uint8)
        >>> obs, _, _, _  = env.step(1)
        >>> obs.shape
        (27648,)

        >>> env = ExampleEnv(observation_space=Dict(left_eye=Box(0, 1, (10, 10, 3)), right_eye=Box(0, 1, (20, 20, 3))))
        >>> env = flatten_observations_v0(env)
        >>> env.observation_space
        Box(0.0, 1.0, (1500,), float32)

    Partially flatten example with composite observation spaces:
        >>> env = ExampleEnv(observation_space=Dict(left_arm=Box(-1, 1, (3, 3)), right_arm=Box(-1, 1, (3, 3))))
        >>> env = flatten_observations_v0(env, {"left_arm": True, "right_arm": False})
        >>> env.observation_space
        Dict(left_arm: Box(-1.0, 1.0, (9,), float32)), right_arm: Box(-1.0, 1.0, (3, 3), float32))
    """

    def __init__(self, env: gym.Env, args: Optional[FuncArgType[bool]] = None):
        """Constructor for flatten observation wrapper.

        Args:
            env: The environment to wrap
            args: The optional flattened arguments
        """
        if args is None:
            flatten_obs_space = flatten_space(env.observation_space)
            func_args = env.observation_space
        else:
            flatten_obs_space = apply_function(
                env.observation_space,
                env.observation_space,
                lambda x, arg: x if arg is False else flatten_space(x),
                args,
            )

            func_args = {}
            for arg, space in zip(args.keys(), flatten_obs_space.values()):
                if args.get(arg, False):
                    func_args[arg] = space

        super().__init__(
            env,
            lambda obs, space: obs if space is None else flatten(space, obs),
            func_args,
            flatten_obs_space,
        )


class grayscale_observations_v0(lambda_observations_v0):
    """A wrapper that converts color observations to grayscale that are returned by :meth:`step` and :meth:`reset`.

    Basic Example with Box observation space:
        >>> import gym
        >>> from gym.spaces import Dict, Box, Discrete
        >>> env = gym.make("CarRacing-v1")
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = grayscale_observations_v0(env)
        >>> env.observation_space
        Box(0, 255, (96, 96), uint8)

    Composite Example with Multiple Box observation space:
        >>> env = gym.vector.make("CarRacing-v1", num_envs=3)
        >>> env.observation_space
        Box(0, 255, (3, 96, 96, 3), uint8)
        >>> env = grayscale_observations_v0(env)
        >>> env.observation_space
        Box(0, 255, (3, 96), uint8)

    Composite Example with Partial Box observation space:
        >>> env = ExampleEnv(observation_space=Dict(obs=Box(0, 255, (96, 96, 3), np.uint8), time=Discrete(10)))
        >>> env.observation_space
        Dict(obs: Box(0, 255, (96, 96, 3), uint8), time: Discrete(10))
        >>> env = grayscale_observations_v0(env, {"obs": True, "time": False})
        >>> env.observation_space
        Dict(obs: Box(0, 255, (96, 96), uint8), time: Discrete(10))
    """

    def __init__(self, env: gym.Env, args: FuncArgType[bool] = True):
        """Constructor for grayscale observation wrapper.

        Args:
            env: The environment to wrap
            args: The arguments for what to convert colour to grayscale in the observation
        """
        observation_space = self._grayscale_space(env, args)

        super().__init__(
            env,
            lambda obs, arg: obs
            if arg is False
            else jp.dot(
                obs[..., :3], jp.array([0.2989, 0.5870, 0.1140])
            ),  # todo, bug in that jp.dot will always return jax.array
            args,
            observation_space,
        )

    def _grayscale_space(self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
        """Process the space and apply the grayscale transformation."""
        return grayscale_space(env.observation_space, args, grayscale_space)


class resize_observations_v0(lambda_observations_v0):
    """A wrapper that resizes an observation returned by :meth:`step` and :meth:`reset` to a new shape.

    Basic Example with Box observation space:
        >>> import gym
        >>> from gym.spaces import Dict, Box, Discrete
        >>> env = gym.make("CarRacing-v1")
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = resize_observations_v0(env, (64, 64, 3))
        >>> env.observation_space
        Box(0, 255, (64, 64, 3), uint8)

    Composite Example with Multiple Box observation space:
        >>> env = gym.vector.make("CarRacing-v1", num_envs=3)
        >>> env.observation_space
        TODO
        >>> env = resize_observations_v0(env, [(64, 64) for _ in range(3)])
        >>> env.observation_space
        TODO

    Composite Example with Partial Box observation space:
        >>> env = ExampleEnv(observation_space=Dict(obs=Box(0, 1, (96, 96, 3)), time=Discrete(10)))
        >>> env.observation_space
        Dict(obs=Box(0, 1, (96, 96, 3)), time=Discrete(10))
        >>> env = resize_observations_v0(env, {"obs": (64, 64, 3)})
        >>> env.observation_space
       Dict(obs=Box(0, 1, (64, 64, 3)), time=Discrete(10))
    """

    def __init__(self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
        """Constructor for resize observation wrapper.

        Args:
            env: The environment to wrap
            args: The arguments to resize the observation
        """
        observation_space = self._resize_space(env, args)

        super().__init__(
            env,
            lambda obs, arg: obs if arg is None else tinyscaler.scale(obs, arg),
            args,
            observation_space,
        )

    def _resize_space(self, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
        """Resize space to args dimension."""
        return resize_space(env.observation_space, args, resize_space)


class reshape_observations_v0(lambda_observations_v0):
    """A wrapper that reshapes an observation returned by :meth:`step` and :meth:`reset` to a new shape.

    Basic Example with Box observation space:
        >>> import gym
        >>> from gym.spaces import Dict, Box, Discrete
        >>> env = gym.make("CarRacing-v1")
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = reshape_observations_v0(env, (96, 36, 8))
        >>> env.observation_space
        Box(0.0, 255.0, (96, 36, 8), uint8)

    Composite Example with Multiple Box observation space:
        >>> env = gym.vector.make("CarRacing-v1", num_envs=3)
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = reshape_observations_v0(env, [(96, 36, 8), (96, 288), (1, 96, 96, 3)])
        >>> env.observation_space
        TODO

    Composite Example with Partial Box observation space:
        >>> env = ExampleEnv(observation_space=Dict(obs=Box(0, 1, (96, 96, 3)), time=Discrete(10)))
        >>> env.observation_space
        Dict(obs: Box(0.0, 1.0, (96, 96, 3), float32), time: Discrete(10))
        >>> env = reshape_observations_v0(env, {"obs": (96, 36, 8)})
        >>> env.observation_space
        Dict(obs: Box(0.0, 1.0, (96, 36, 8), float32), time: Discrete(10))
    """

    def __init__(self, env: gym.Env, args: FuncArgType[TypingTuple[int, ...]]):
        """Constructor for reshape observation wrapper.

        Args:
            env: The environment to wrap
            args: The arguments to reshape the observation
        """
        observation_space = self._reshape_space(env, args)

        super().__init__(
            env,
            lambda obs, arg: obs if arg is None else jp.reshape(obs, arg),
            args,
            observation_space,
        )


# class observations_dtype_v0(lambda_observations_v0):
#     """A wrapper that converts the observation dtype returned by :meth:`step` and :meth:`reset` to a new shape.

#     Basic Example:
#         >>> import gym
#         >>> from gym.spaces import Dict, Box, Discrete
#         >>> env = gym.make("CartPole-v1")
#         >>> env.observation_space
#         TODO
#         >>> env = observations_dtype_v0(env, jp.float64)
#         >>> env.observation_space
#         TODO

#     Composite Example:
#         >>> env = ExampleEnv(observation_space=Dict())
#         >>> env = observations_dtype_v0(env, TODO)
#         >>> env.observation_space
#         TODO

#         >>> env = ExampleEnv(observation_space=Tuple())
#         >>> env = observations_dtype_v0(env, TODO)
#         >>> env.observation_space
#         TODO

#         >>> env = ExampleEnv(observation_space=Dict(Tuple()))
#         >>> env = observations_dtype_v0(env, TODO)
#         >>> env.observation_space
#         TODO
#     """

#     def __init__(
#         self, env: gym.Env, args: FuncArgType[Union[jp.dtype, str]]
#     ):
#         """Constructor for observation dtype wrapper.

#         Args:
#             env: The environment to wrap
#             args: The arguments for the dtype changes
#         """
#         observation_space = apply_function(env.observation_space, env.observation_space,
#                                            lambda x, arg: setattr(x, 'dtype', arg), args)

#         super().__init__(env, lambda obs, arg: jp.astype(obs, arg), args, observation_space)
