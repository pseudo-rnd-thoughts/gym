import pytest

from gym.wrappers import (
    filter_observations_v0,
    flatten_observations_v0,
    grayscale_observations_v0,
    lambda_observations_v0,
    normalize_observations_v0,
    observation_dtype_v0,
    observations_reshape_v0,
    resize_observations_v0,
)

image_testing_environments = []

all_testing_environments = []


@pytest.mark.parametrize()
def test_lambda_observations_v0(env):
    wrapped_env = lambda_observations_v0(env)


@pytest.mark.parametrize()
def test_filter_observations_v0(env):
    wrapped_env = filter_observations_v0(env)


@pytest.mark.parametrize
def test_flatten_observations_v0(env):
    wrapped_env = flatten_observations_v0(env)


@pytest.mark.parametrize()
def test_grayscale_observations_v0(env):
    wrapped_env = grayscale_observations_v0(env)


@pytest.mark.parametrize()
def test_normalize_observations_v0(env):
    wrapped_env = normalize_observations_v0(env)


@pytest.mark.parametrize()
def test_resize_observations_v0(env):
    wrapped_env = resize_observations_v0(env)


@pytest.mark.parametrize()
def test_observation_dtype_v0(env):
    wrapped_env = observation_dtype_v0(env)


@pytest.mark.parametrize()
def observation_reshape_v0(env):
    wrapped_env = observations_reshape_v0(env)
