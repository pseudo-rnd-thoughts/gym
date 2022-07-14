import dataclasses
import inspect
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct

import gym
from gym import Space, logger, spaces
from gym.core import ActType, ObsType
from gym.spaces.space import T_cov
from gym.utils import seeding

StateType = TypeVar("StateType")
RngType = TypeVar("RngType")


@dataclasses.dataclass
class FunctionalEnv(Generic[ObsType, ActType, StateType, RngType]):
    observation_space: spaces.Space
    action_space: spaces.Space

    initial_state_fn: Callable[
        [RngType, Optional[Dict[str, Any]]],
        Tuple[StateType, RngType, ObsType, Dict[str, Any]],
    ]
    state_transition_fn: Callable[
        [StateType, ActType, RngType],
        Tuple[StateType, RngType, ObsType, Any, Any, Any, Dict[str, Any]],
    ]


JaxState = struct.dataclass


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


class JaxEnv(gym.Env[ObsType, ActType]):
    def __init__(
        self,
        functional_env: FunctionalEnv[JaxState, ObsType, ActType, jnp.DeviceArray],
        jit_fn: bool = True,
        backend: Optional[str] = None,
        static_reset_options: bool = True,
    ):
        assert isinstance(functional_env.observation_space, JaxSpace)
        assert isinstance(functional_env.action_space, JaxSpace)
        self.observation_space: JaxSpace = functional_env.observation_space
        self.action_space: JaxSpace = functional_env.action_space

        initial_state_signature = inspect.signature(functional_env.initial_state_fn)
        if "self" in initial_state_signature.parameters:
            logger.warn(
                "The reset function contains the argument `self`, we recommend that the reset function is stateless and does not include `self` for optimisation."
            )
        state_transition_signature = inspect.signature(
            functional_env.state_transition_fn
        )
        if "self" in state_transition_signature.parameters:
            logger.warn(
                "The step function contains the argument `self`, we recommend that the step function is stateless and does not include `self` for optimisation."
            )

        if jit_fn:
            if static_reset_options:
                self.reset_fn = jax.jit(
                    functional_env.initial_state_fn, static_argnums=1, backend=backend
                )
            else:
                self.reset_fn = jax.jit(
                    functional_env.initial_state_fn, backend=backend
                )
            self.step_fn = jax.jit(functional_env.state_transition_fn, backend=backend)
        else:
            self.reset_fn = functional_env.initial_state_fn
            self.step_fn = functional_env.state_transition_fn

        self.state: Optional[JaxState] = None
        _, seed = seeding.np_random()
        self._np_random: jnp.DeviceArray = jax.random.PRNGKey(
            seed % onp.iinfo(onp.int64).max
        )

        self._is_closed = False

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
            self.np_random = jax.random.PRNGKey(seed)

        self.state, self.np_random, obs, info = self.reset_fn(self.np_random, options)

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
        ) = self.step_fn(self.state, action, self.np_random)

        return obs, reward, terminated, truncated, info

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
