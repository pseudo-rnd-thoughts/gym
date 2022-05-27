"""Module of wrapper classes."""
from gym.dev_wrappers.lambda_action import (
    clip_actions_v0,
    lambda_action_v0,
    scale_actions_v0,
)
from gym.dev_wrappers.lambda_observations import (
    filter_observations_v0,
    flatten_observations_v0,
    grayscale_observations_v0,
    lambda_observations_v0,
    observations_dtype_v0,
    reshape_observations_v0,
    resize_observations_v0,
)
from gym.dev_wrappers.lambda_reward import (
    clip_rewards_v0,
    lambda_reward_v0,
)
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.autoreset import AutoResetWrapper
from gym.wrappers.clip_action import ClipAction
from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.frame_stack import FrameStack, LazyFrames
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.normalize import NormalizeObservation, NormalizeReward
from gym.wrappers.order_enforcing import OrderEnforcing
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.record_video import RecordVideo, capped_cubic_video_schedule
from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.time_aware_observation import TimeAwareObservation
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers.transform_observation import TransformObservation
from gym.wrappers.transform_reward import TransformReward
