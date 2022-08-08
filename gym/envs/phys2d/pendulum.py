"""
Implementation of a Jax-accelerated pendulum environment.
"""
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import numpy as np

from gym.envs.jax_env import JaxSpace
from gym.functional import ActType, FunctionalEnv, StateType
from gym.spaces import Box


class FunctionalPendulum(
    FunctionalEnv[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """A functional implementation of classic control Pendulum in Jax."""

    max_speed = 8
    max_torque = 2.0
    dt = 0.05
    gravity = 10.0
    mass = 1.0
    length = 1.0
    max_x = jnp.pi
    max_y = 1.0

    def __init__(self, options: Dict[str, Any] = None):
        if options is not None:
            for name, value in options.items():
                setattr(self, name, value)

        super().__init__(
            observation_space=JaxSpace(Box(-np.inf, np.inf, shape=(4,))),
            action_space=JaxSpace(Box(-self.max_torque, self.max_torque)),
        )

    def initial(self, rng: jnp.ndarray):
        """Initial state generation."""
        high = jnp.array([self.max_x, self.max_y])
        return jax.random.uniform(key=rng, minval=-high, maxval=high, shape=high.shape)

    def transition(
        self, state: jnp.ndarray, action: Union[int, jnp.ndarray], rng: None = None
    ) -> jnp.ndarray:
        """Pendulum transition."""
        theta, theta_dot = state
        u = jnp.clip(action, -self.max_torque, self.max_torque)[0]

        new_theta_dot = (
            theta_dot
            + (
                3 * self.gravity / (2 * self.length) * jnp.sin(theta)
                + 3.0 / (self.mass * self.length**2) * u
            )
            * self.dt
        )
        new_theta_dot = jnp.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * self.dt

        return jnp.array([new_theta, new_theta_dot])

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        theta, theta_dot = state
        return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> jnp.ndarray:
        theta, theta_dot = next_state  # todo - to confirm, next state not state
        u = jnp.clip(action, -self.max_torque, self.max_torque)[0]

        th_normalized = ((theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        costs = th_normalized**2 + 0.1 * theta_dot**2 + 0.001 * (u**2)

        return -costs

    def terminal(self, state: StateType) -> jnp.ndarray:
        return jnp.zeros((), dtype=jnp.bool_)

    def truncate(self, state: StateType) -> jnp.ndarray:
        return jnp.zeros((), dtype=jnp.bool_)

    def information(self, state: StateType) -> Dict[str, jnp.ndarray]:
        return {"test": jnp.zeros(1)}
