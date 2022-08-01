from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

import gym
from gym import Space
from gym.core import ActType, ObsType
from gym.functional import FuncEnv
from gym.spaces.space import T_cov
from gym.utils import seeding

StateType = TypeVar("StateType")
RngType = TypeVar("RngType")

JaxState = struct.dataclass


def jax_to_numpy(value):
    pass


def numpy_to_jax(value):
    pass


class JaxSpace(Space[Any]):
    def __init__(self, space):
        self.space = space

        dtype = jnp.dtype(space.dtype) if space.dtype is not None else None
        super().__init__(space.shape, dtype, space.np_random)

    def sample(self, mask: Optional[Any] = None) -> T_cov:
        space_sample = self.space.sample(mask)

        return numpy_to_jax(space_sample)

    def contains(self, x) -> bool:
        numpy_x = jax_to_numpy(x)

        return self.space.contains(numpy_x)

    def seed(self, seed: Optional[int] = None) -> list:
        return self.space.seed(seed)

    def to_jsonable(self, sample_n: Sequence[T_cov]) -> list:
        return self.space.to_jsonable(jax_to_numpy(sample_n))

    def from_jsonable(self, sample_n: list) -> List[T_cov]:
        return [numpy_to_jax(value) for value in self.space.from_jsonable(sample_n)]


class JaxEnv(gym.Env[ObsType, ActType]):
    def __init__(
        self,
        func_env: FuncEnv[JaxState, ObsType, ActType, jnp.DeviceArray],
        jit_funcs: bool = True,
        backend: Optional[str] = None,
    ):
        # Expectation that jax environment will have a jax-based space
        assert isinstance(func_env.observation_space, JaxSpace)
        assert isinstance(func_env.action_space, JaxSpace)

        # If enabled, transform the environment with `jax.jit`
        if jit_funcs:
            func_env.transform(partial(jax.jit, backend=backend))

        # Initialise class variables
        self.env = func_env
        self.state: Optional[JaxState] = None
        self._np_random: jnp.DeviceArray = jax.random.PRNGKey(
            seeding.np_random()[1] % onp.iinfo(onp.int64).max
        )

        # For rendering, to ensure that pygame is closed when the environment is deleted correctly.
        self._is_closed = False

    @property
    def np_random(self) -> jnp.DeviceArray:
        return self._np_random

    @np_random.setter
    def np_random(self, value: jnp.DeviceArray):
        self._np_random = value

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
        self.state = self.env.initial(initial_rng)

        if return_info:
            return self.env.observation(self.state), self.env.info(self.state)
        else:
            return self.env.observation(self.state)

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray, Dict[str, Any]
    ]:
        self.np_random, transition_rng = jax.random.split(self.np_random)

        next_state = self.env.transition(self.state, action, transition_rng)

        obs = self.env.observation(next_state)
        reward = self.env.reward(self.state, action, next_state)
        terminated = self.env.terminal(next_state)
        info = self.env.info(next_state)

        # todo: truncation
        truncation = jnp.zeros(1)

        self.state = next_state
        return obs, reward, terminated, truncation, info

    def close(self):
        self._is_closed = True

    def __del__(self):
        if getattr(self, "_is_close", False):
            self.close()


class VectorizeJaxEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        env: JaxEnv,
        num_envs: int,
        device_parallelism: bool = False,
    ):
        super().__init__(num_envs, env.observation_space.space, env.action_space.space)
        self.env = env

        self.observation_space = JaxSpace(self.observation_space)
        self.action_space = JaxSpace(self.action_space)

        if device_parallelism:
            self.vectorise_reset_fn = jax.pmap(
                env.reset_fn,
                in_axes=[0, None],
                static_broadcasted_argnums=1,
                axis_name="gym-reset",
                axis_size=num_envs,
            )
            self.vectorise_step_fn = jax.pmap(
                env.step_fn,
                in_axes=[0, 0, 0],
                axis_name="gym-step",
                axis_size=num_envs,
            )
        else:
            self.vectorise_reset_fn = jax.vmap(
                env.reset_fn,
                in_axes=[0, None],
                axis_name="gym-reset",
                axis_size=num_envs,
            )
            self.vectorise_step_fn = jax.vmap(
                env.step_fn,
                in_axes=[0, 0, 0],
                axis_name="gym-step",
                axis_size=num_envs,
            )

        self.state: Optional[JaxState] = None
        _, seed = seeding.np_random()
        self._np_random: jnp.DeviceArray = jax.random.split(
            jax.random.PRNGKey(seed % onp.iinfo(onp.int64).max), num_envs
        )

    @property
    def np_random(self) -> jnp.DeviceArray:
        return self._np_random

    @np_random.setter
    def np_random(self, value):
        self._np_random = value

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        if seed is not None:
            self.np_random = jax.random.split(jax.random.PRNGKey(seed), self.num_envs)

        self.state, self.np_random, obs, info = self.vectorise_reset_fn(
            self.np_random, options
        )

        if return_info:
            return obs, info
        else:
            return obs

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray, Dict[str, Any]
    ]:
        (
            self.state,
            self.np_random,
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.vectorise_step_fn(self.state, action, self.np_random)

        return obs, reward, terminated, truncated, info

    def __repr__(self) -> str:
        """Returns a string representation of the vector environment.

        Returns:
            A string containing the class name, number of environments and environment spec id
        """
        if self.env.spec is None:
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"


def jax_func_env_checker(env: FuncEnv):

    # Chex max_compile
    # observation type
    # action type
    # function signatures

    pass
