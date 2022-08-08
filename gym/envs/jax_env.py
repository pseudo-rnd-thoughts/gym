from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

import gym
from gym import Space
from gym.core import ActType, ObsType
from gym.functional import FunctionalEnv
from gym.spaces.space import T_cov
from gym.utils import seeding

StateType = TypeVar("StateType")
RngType = TypeVar("RngType")

JaxState = struct.dataclass


def jax_to_numpy(value):
    return onp.array(value)


def numpy_to_jax(value):
    return jnp.array(value)


class JaxSpace(Space[Any]):
    """
    To allow compatibility between Jax and gym.Space, this class is a wrapper for gym.Spaces to work with jax arrays

    A limitation is this requires numpy sampling

    TODO - convert all gym spaces to jumpy to allow completely jitted workflows.
    """

    def __init__(self, space):
        self.space = space

        dtype = jnp.dtype(space.dtype) if space.dtype is not None else None
        super().__init__(space.shape, dtype, space.np_random)

    def sample(self, mask: Optional[Any] = None) -> T_cov:
        space_sample = self.space.sample(mask)

        return numpy_to_jax(space_sample)

    def contains(self, x) -> bool:
        """Run the `self.space.contains` using a numpy version of `x`.

        Assuming that `x` is a jax array and `self.space` expects a numpy array.
        We convert from jax to numpy then run contains.
        """
        numpy_x = jax_to_numpy(x)

        return self.space.contains(numpy_x)

    def seed(self, seed: Optional[int] = None) -> list:
        """Seeds the space.

        TODO - Convert the seed such that the spaces use jax.random"""
        return self.space.seed(seed)

    def __getattr__(self, item):
        """Gets the attribute of the"""
        if item == "dtype":
            return self.dtype
        else:
            return getattr(self.space, item)

    def to_jsonable(self, sample_n: Sequence[T_cov]) -> list:
        """Runs the spaces `to_jsonable` function with a numpy version of sample_n."""
        return self.space.to_jsonable(jax_to_numpy(sample_n))

    def from_jsonable(self, sample_n: list) -> List[T_cov]:
        """Runs the spaces `from_jsonable` function that is converted back to jax."""
        return [numpy_to_jax(value) for value in self.space.from_jsonable(sample_n)]


class JaxEnv(gym.Env[ObsType, ActType]):
    """A interface for jax-based FuncEnv to appear like a gym.Env.

    TODO - Add rendering
    """

    metadata = {
        # todo - ??
    }

    def __init__(
        self,
        func_env: FunctionalEnv,
        jit_funcs: bool = True,
        backend: Optional[str] = None,
    ):
        # Expectation that jax environment will have a jax-based space
        assert isinstance(func_env.observation_space, JaxSpace)
        assert isinstance(func_env.action_space, JaxSpace)

        self.observation_space: JaxSpace = JaxSpace(func_env.observation_space)
        self.action_space: JaxSpace = JaxSpace(func_env.action_space)

        # If enabled, transform the environment with `jax.jit`
        if jit_funcs:
            func_env.transform(
                partial(jax.jit, static_argnames=("self",), backend=backend)
            )

        # Initialise class variables
        self.func_env = func_env
        self.state: Optional[JaxState] = None
        self._np_random: jnp.ndarray = jax.random.PRNGKey(
            seeding.np_random()[1] % onp.iinfo(onp.int64).max
        )

        # For rendering, to ensure that pygame is closed when the environment is deleted correctly.
        self._is_closed = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        if seed is not None:
            self.np_random = jax.random.PRNGKey(seed)

        self.np_random, initial_rng = jax.random.split(self.np_random)
        self.state = self.func_env.initial(initial_rng)

        return (
            self.func_env.observation(self.state),
            self.func_env.information(self.state),
        )

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        self.np_random, transition_rng = jax.random.split(self.np_random)
        next_state = self.func_env.transition(self.state, action, transition_rng)

        obs = self.func_env.observation(next_state)
        reward = self.func_env.reward(self.state, action, next_state)
        terminated = self.func_env.terminal(next_state)
        truncation = self.func_env.truncate(next_state)
        info = self.func_env.information(next_state)

        self.state = next_state
        return obs, reward, terminated, truncation, info

    def close(self):
        """Close the environment."""
        self._is_closed = True

    def __del__(self):
        """On destruction, closes the environment if not already done."""
        if getattr(self, "_is_close", False):
            self.close()

    def __repr__(self) -> str:
        return self.func_env.__class__.__name__


class VectorizeJaxEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        env: FunctionalEnv,
        num_envs: int,
        device_parallelism: bool = False,
        backend: Optional[str] = None,
    ):
        super().__init__(num_envs, env.observation_space, env.action_space)
        self.func_env = env

        self.observation_space = JaxSpace(self.observation_space)
        self.action_space = JaxSpace(self.action_space)

        for func_name, in_axes in [
            ("initial", (None, 0, 0)),
            ("transition", (None, 0, 0, 0)),
            ("observation", (None, 0)),
            ("reward", (None, 0, 0, 0)),
            ("terminal", (None, 0)),
            ("truncate", (None, 0)),
            ("info", (None, 0, 0, 0)),
        ]:
            if device_parallelism:
                setattr(
                    env,
                    func_name,
                    jax.pmap(
                        getattr(env, func_name),
                        in_axes=in_axes,
                        axis_name=f"gym-{func_name}",
                        axis_size=num_envs,
                        static_broadcasted_argnums=1,
                        backend=backend,
                    ),
                )
            else:
                setattr(
                    env,
                    func_name,
                    jax.vmap(
                        getattr(env, func_name),
                        in_axes=in_axes,
                        axis_name=f"gym-{func_name}",
                        axis_size=num_envs,
                    ),
                )

        self.state: Optional[JaxState] = None
        self._np_random: jnp.ndarray = jax.random.PRNGKey(
            seeding.np_random()[1] % onp.iinfo(onp.int64).max
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        if seed is not None:
            self.np_random = jax.random.PRNGKey(seed)

        self.np_random, initial_rng = jax.random.split(self.np_random)
        vec_initial_rng = jax.random.split(initial_rng, num=self.num_envs)
        self.state = self.func_env.initial(vec_initial_rng)

        return (
            self.func_env.observation(self.state),
            self.func_env.information(self.state),
        )

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        self.np_random, transition_rng = jax.random.split(self.np_random)
        vec_transition_rng = jax.random.split(transition_rng, num=self.num_envs)
        next_state = self.func_env.transition(self.state, action, vec_transition_rng)

        obs = self.func_env.observation(next_state)
        reward = self.func_env.reward(self.state, action, next_state)
        terminated = self.func_env.terminal(next_state)
        truncation = self.func_env.truncate(next_state)
        info = self.func_env.information(next_state)

        self.state = next_state
        return obs, reward, terminated, truncation, info

    def __repr__(self) -> str:
        """Returns a string representation of the vector environment.

        Returns:
            A string containing the class name, number of environments and environment spec id
        """
        return f"{self.func_env.__class__.__name__}({self.num_envs})"


def jax_env_checker(env: FunctionalEnv):
    """
    A specialised implementation of the gym environment checker just for functional jax environment.

    Use the gym environment checker as a `JaxEnv(env)` to ensure that the environment follows the rest of the gym api.

    This checks
    1. observation and action types - Check that the observation and action spaces are jax spaces to allow sampling and contains
    2. jax jit compiling - on all stateless functions, the number of times the function is
        jax.jit max compile is 1.
    """
    assert isinstance(env, FunctionalEnv), type(env)

    assert isinstance(env.observation_space, JaxSpace)
    assert isinstance(env.action_space, JaxSpace)

    # First wrap the environment with `chex.assert_max_traces`
    env.transform(partial(chex.assert_max_traces, n=1))
    env.transform(partial(jax.jit, static_argnames=("self",)))

    rng = jax.random.PRNGKey(1)
    for _ in range(3):
        rng, initial_rng = jax.random.split(rng)
        state = env.initial(initial_rng)

        obs = env.observation(state)
        info = env.information(state)
        print(f"obs={obs}, info={info}")

        for _ in range(10):
            action = env.action_space.sample()
            rng, transition_rng = jax.random.split(rng)

            next_state = env.transition(state, action, transition_rng)

            obs = env.observation(next_state)
            reward = env.reward(state, action, next_state)
            termination = env.terminal(next_state)
            truncation = env.truncate(next_state)
            info = env.information(next_state)
            print(
                f"obs={obs}, reward={reward}, termination={termination}, truncation={truncation}, info={info}"
            )

            state = next_state

            if termination or truncation:
                break
