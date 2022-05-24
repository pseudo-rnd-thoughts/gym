

import gym
for env_spec in gym.envs.registry.values():
    env = env_spec.make()
    print(f'{env_spec.id} - {env.observation_space=}')
