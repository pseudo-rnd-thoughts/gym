"""Utility functions for gym environments."""
from __future__ import annotations

from typing import Union

import numpy as np

from gym.utils.seeding import RandomNumberGenerator


def categorical_sample(
    prob_n: Union[list, np.ndarray], np_random: RandomNumberGenerator
) -> np.ndarray:
    """Samples from categorical distribution with each row specifies class probabilities.

    Args:
        prob_n: Array of probabilities for each class
        np_random: Environment random number generator

    Returns:
        Categorical samples
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return np.argmax(csprob_n > np_random.random())
