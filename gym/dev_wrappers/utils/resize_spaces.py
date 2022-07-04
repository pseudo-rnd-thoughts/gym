"""A set of utility functions for lambda wrappers."""
from functools import singledispatch
from typing import Any, Callable
from typing import Tuple as TypingTuple

import tinyscaler

from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space


@singledispatch
def resize_space(
    space: Space, args: FuncArgType[TypingTuple[int, int]], fn: Callable
) -> Any:
    """Resize space with the provided args."""


@resize_space.register(Box)
def _resize_space_box(space, args: FuncArgType[TypingTuple[int, int]], fn: Callable):
    if args is not None:
        return Box(
            tinyscaler.scale(space.low, args, mode='bilinear'),
            tinyscaler.scale(space.high, args, mode='bilinear'),
            shape=args,
            dtype=space.dtype,
        )
    return space


@resize_space.register(Discrete)
@resize_space.register(MultiBinary)
@resize_space.register(MultiDiscrete)
def _reshape_space_not_reshapable(
    space, args: FuncArgType[TypingTuple[int, int]], fn: Callable
):
    """Return original space shape for not reshable space.

    Trying to reshape `Discrete`, `Multibinary` and `MultiDiscrete`
    spaces has no effect.
    """
    if args:
        # TODO: raise warning that args has no effect here
        ...
    return space
