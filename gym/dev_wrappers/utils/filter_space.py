"""A set of utility functions for lambda wrappers."""
from functools import singledispatch
from typing import Any, Callable
from typing import Tuple as TypingTuple

from gym.dev_wrappers import FuncArgType
from gym.spaces import Dict, Space, Tuple


@singledispatch
def filter_space(
    space: Space, args: FuncArgType[TypingTuple[int, int]], fn: Callable
) -> Any:
    """Filter space with the provided args."""


@filter_space.register(Dict)
def _filter_space_dict(
    space: Space, args: FuncArgType[TypingTuple[int, int]], fn: Callable
):
    """Filter `Dict` observation space by args."""
    if isinstance(args, list):
        args = {arg: True for arg in args}
    return (
        Dict([(name, value) for name, value in space.items() if args.get(name, False)]),
        args,
    )


@filter_space.register(Tuple)
def _filter_space_tuple(
    space: Space, args: FuncArgType[TypingTuple[int, int]], fn: Callable
):
    """Filter `Tuple` observation space by args."""
    if len(args) != len(space):  # we are passing args as indexes
        args = [True if i in args else False for i in range(len(space))]
    return Tuple([value for value, arg in zip(space, args) if arg]), args
