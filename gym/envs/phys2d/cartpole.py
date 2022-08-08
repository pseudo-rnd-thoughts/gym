"""
Implementation of a Jax-accelerated cartpole environment.
"""

from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

import gym
from gym.envs.jax_env import JaxSpace
from gym.functional import ActType, FunctionalEnv, StateType
from gym.spaces import Box


class FunctionalCartPole(
    FunctionalEnv[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """A functional implementation of classic control CartPole in Jax."""

    gravity = jnp.array(9.8)
    cart_mass = jnp.array(1.0)
    pole_mass = jnp.array(0.1)
    length = jnp.array(0.5)
    force_mag = jnp.array(10.0)
    dt = jnp.array(0.02)
    theta_threshold_radians = jnp.array(12 * 2 * jnp.pi / 360)
    x_threshold = jnp.array(2.4)
    x_init = jnp.array(0.05)

    def __init__(self, options: Dict[str, Any] = None):
        if options is not None:
            for name, value in options.items():
                assert getattr(self, name) is not None
                setattr(self, name, value)

        self.total_mass = self.pole_mass + self.cart_mass
        self.pole_mass_length = self.pole_mass + self.length

        super().__init__(
            observation_space=JaxSpace(Box(-np.inf, np.inf, shape=(4,))),
            action_space=JaxSpace(gym.spaces.Discrete(2)),
        )

    def initial(self, rng: PRNGKey):
        """Initial state generation."""
        return jax.random.uniform(
            key=rng, minval=-self.x_init, maxval=self.x_init, shape=(4,)
        )

    def transition(
        self, state: jnp.ndarray, action: Union[int, jnp.ndarray], rng: None = None
    ) -> StateType:
        """Cartpole transition."""
        x, x_dot, theta, theta_dot = state  # todo - investigate performance
        force = jnp.sign(action - 0.5) * self.force_mag
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.pole_mass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.pole_mass * costheta**2 / self.total_mass)
        )
        xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * xacc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * thetaacc

        state = jnp.array((x, x_dot, theta, theta_dot), dtype=jnp.float32)

        return state

    def observation(self, state: jnp.ndarray) -> jnp.ndarray:
        """Cartpole observation."""
        return state

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> jnp.ndarray:
        x, _, theta, _ = state

        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )
        return jnp.logical_not(terminated).astype(jnp.int32)

    def terminal(self, state: jnp.ndarray) -> jnp.ndarray:
        x, _, theta, _ = state

        return (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

    def truncate(self, state: StateType) -> jnp.ndarray:
        return jnp.zeros((), dtype=jnp.bool_)

    def information(self, state: StateType) -> Dict[str, jnp.ndarray]:
        return {}
