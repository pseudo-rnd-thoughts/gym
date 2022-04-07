from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from gym import logger

from .discrete import Discrete
from .space import Space


class MultiDiscrete(Space[np.ndarray]):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different number of actions in each
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of positive integers specifying number of actions for each discrete action space

    Note: Some environment wrappers assume a value of 0 always represents the NOOP action.

    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:

        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

    - Can be initialized as

        MultiDiscrete([ 5, 2, 2 ])

    """

    def __init__(
        self,
        nvec: list[int] | np.ndarray,
        dtype: np.dtype = np.int64,
        seed: Optional[int] = None,
    ):
        """
        nvec: vector of counts of each categorical variable
        """
        self.nvec = np.array(nvec, dtype=dtype, copy=True)
        assert (self.nvec > 0).all(), "nvec (counts) have to be positive"

        super().__init__(self.nvec.shape, dtype, seed)

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape  # type: ignore

    def sample(self) -> np.ndarray:
        """Randomly sample an element of this space."""
        return (self.np_random.random(self.nvec.shape) * self.nvec).astype(self.dtype)

    def contains(self, x: Sequence | np.ndarray) -> bool:
        """Return boolean specifying if x is a valid member of this space"""
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return bool(x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all())

    def to_jsonable(self, sample_n: Sequence[np.ndarray]) -> list[list]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n: list) -> np.ndarray:
        """Convert a JSONable data type to a batch of samples from this space."""
        return np.array(sample_n)

    def __repr__(self):
        return f"MultiDiscrete({self.nvec})"

    def __getitem__(self, index: int):
        nvec = self.nvec[index]
        if nvec.ndim == 0:
            subspace = Discrete(nvec)
        else:
            subspace = MultiDiscrete(nvec, self.dtype)
        subspace.np_random.bit_generator.state = self.np_random.bit_generator.state
        return subspace

    def __len__(self):
        if self.nvec.ndim >= 2:
            logger.warn("Get length of a multi-dimensional MultiDiscrete space.")
        return len(self.nvec)

    def __eq__(self, other):
        return isinstance(other, MultiDiscrete) and np.all(self.nvec == other.nvec)
