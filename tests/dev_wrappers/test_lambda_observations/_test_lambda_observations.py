# """

# """
# import gym
# import pytest

# from gym.spaces import Dict, Tuple, Box, MultiDiscrete, MultiBinary
# from gym.wrappers import (
#     filter_observations_v0,
#     flatten_observations_v0,
#     grayscale_observations_v0,
#     lambda_observations_v0,
#     observations_dtype_v0,
#     reshape_observations_v0,
#     resize_observations_v0,
# )
# from tests.dev_wrappers.utils import contains_space, TestingEnv

# NUM_ENVS = 3
# ENVS = [
#     gym.make('CartPole-v1'),    # Box(np.zeros(4), np.ones(4), (4,), float32)
#     gym.make('CarRacing-v1'),   # Box(0, 255, (96, 96, 3), uint8)
#     gym.make('FrozenLake-v1'),  # Discrete(16)
#     gym.make('Blackjack-v1'),   # Tuple(Discrete(32), Discrete(11), Discrete(2))
#     gym.vector.make('CartPole-v1', num_envs=NUM_ENVS),
#     gym.vector.make('CarRacing-v1', num_envs=NUM_ENVS),
#     gym.vector.make('FrozenLake-v1', num_envs=NUM_ENVS),
#     gym.vector.make('Blackjack-v1', num_envs=NUM_ENVS),
#     TestingEnv(observation_space=MultiDiscrete([5, 3])),
#     TestingEnv(observation_space=MultiBinary(10)),
#     TestingEnv(observation_space=MultiBinary([3, 4])),
#     TestingEnv(observation_space=Dict()),
#     TestingEnv(observation_space=Dict()),
#     TestingEnv(observation_space=Tuple([]))
# ]
# IMAGE_ENVS = [env for env in ENVS if contains_space(env.observation_space, Box)]
# COMPOSITE_ENVS = [env for env in ENVS if isinstance(env.observation_space, (Dict, Tuple))]

# SEED = 1
# NUM_STEPS = 5


# @pytest.mark.parametrize("env", ENVS)
# @pytest.mark.parametrize("func,args,updated_obs_shape", [

# ])
# def test_lambda_observations_v0(env, func, args, updated_obs_shape):
#     wrapped_env = lambda_observations_v0(env, func, args, updated_obs_shape)

#     wrapped_env.reset(seed=SEED)
#     for _ in range(NUM_STEPS):
#         action = wrapped_env.action_space.sample()
#         obs, *res = wrapped_env.step(action)
#         assert obs in wrapped_env.observation_space


# @pytest.mark.parametrize("env", COMPOSITE_ENVS, ids=[])
# @pytest.mark.parametrize("args", [

# ])
# def test_filter_observations_v0(env, args):
#     wrapped_env = filter_observations_v0(env, args)

#     wrapped_env.reset(seed=SEED)
#     for _ in range(NUM_STEPS):
#         action = wrapped_env.action_space.sample()
#         obs, *res = wrapped_env.step(action)
#         assert obs in wrapped_env.observation_space


# @pytest.mark.parametrize("env", ENVS)
# @pytest.mark.parametrize("args", [

# ])
# def test_flatten_observations_v0(env, args):
#     wrapped_env = flatten_observations_v0(env, args)

#     wrapped_env.reset(seed=SEED)
#     for _ in range(NUM_STEPS):
#         action = wrapped_env.action_space.sample()
#         obs, *res = wrapped_env.step(action)
#         assert obs in wrapped_env.observation_space


# @pytest.mark.parametrize("env", IMAGE_ENVS)
# @pytest.mark.parametrize("args", [

# ])
# def test_grayscale_observations_v0(env, args):
#     wrapped_env = grayscale_observations_v0(env, args)

#     wrapped_env.reset(seed=SEED)
#     for _ in range(NUM_STEPS):
#         action = wrapped_env.action_space.sample()
#         obs, *res = wrapped_env.step(action)
#         assert obs in wrapped_env.observation_space


# @pytest.mark.parametrize("env", IMAGE_ENVS)
# @pytest.mark.parametrize("args", [

# ])
# def test_resize_observations_v0(env, args):
#     wrapped_env = resize_observations_v0(env, args)

#     wrapped_env.reset(seed=SEED)
#     for _ in range(NUM_STEPS):
#         action = wrapped_env.action_space.sample()
#         obs, *res = wrapped_env.step(action)
#         assert obs in wrapped_env.observation_space


# @pytest.mark.parametrize("env", ENVS)
# @pytest.mark.parametrize("args", [

# ])
# def test_observation_dtype_v0(env, args):
#     wrapped_env = observations_dtype_v0(env, args)

#     wrapped_env.reset(seed=SEED)
#     for _ in range(NUM_STEPS):
#         action = wrapped_env.action_space.sample()
#         obs, *res = wrapped_env.step(action)
#         assert obs in wrapped_env.observation_space


# @pytest.mark.parametrize("env", ENVS)
# @pytest.mark.parametrize("args", [

# ])
# def test_reshape_observation_v0(env, args):
#     wrapped_env = reshape_observations_v0(env, args)

#     wrapped_env.reset(seed=SEED)
#     for _ in range(NUM_STEPS):
#         action = wrapped_env.action_space.sample()
#         obs, *res = wrapped_env.step(action)
#         assert obs in wrapped_env.observation_space

