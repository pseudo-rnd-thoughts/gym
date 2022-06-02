import gym
from gym import logger

spec_list = []
for spec in gym.envs.registry.values():
    try:
        spec.make(disable_env_checker=True)
        spec_list.append(spec)
    except Exception as e:
        logger.warn(f"{spec.id} not added to spec list due to exception: {e}")

spec_list_no_mujoco_py = [
    spec for spec in spec_list if "gym.envs.mujoco" not in spec.entry_point
]
