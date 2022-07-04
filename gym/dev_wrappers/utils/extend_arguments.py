"""A set of utility functions for lambda wrappers."""
import gym
from copy import deepcopy
from functools import singledispatch
from typing import Callable, Sequence, Union
from typing import Tuple as TypingTuple
from gym.dev_wrappers import FuncArgType

from gym.spaces import Dict, Space, Tuple, Box


def is_iterable_args(args: Union[list, dict, tuple]):
    return isinstance(args, list) or isinstance(args, dict)


@singledispatch
def extend_nestable_args(space, updated_space, i, args, fn):
    ...


@singledispatch
def extend_args(space: Space, args: dict, fn: Callable):
    ...


@extend_args.register(Box)
def _extend_args_box(space: Space, args: Sequence, fn: Callable):
    if args is None:
        return (space.low, space.high, space.low, space.high)
    #  For asymmetrical spaces?
    return (*args, space.low.min(), space.high.max())


@extend_args.register(Tuple)
def _extend_args_tuple(
    space: Tuple, args: FuncArgType[TypingTuple[int, int]], fn: Callable
):
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    extended_args = [arg for arg in args]

    for i, arg in enumerate(args):
        if is_iterable_args(arg):
            extend_nestable_args(space[i], extended_args, i, args[i], fn)
        else:
            extended_args[i] = fn(space[i], args[i], fn)
    return extended_args


@extend_args.register(Dict)
def _extend_args_dict(space: Tuple, args: FuncArgType[TypingTuple[int, int]], fn: Callable):
    """Extend args for rescaling actions.

    Action space args needs to be extended in order
    to correctly rescale the actions.
    i.e. args before: {"body":{"left_arm": (-0.5,0.5)}, ...}
    args after: {"body":{"left_arm": (-0.5,0.5,-1,1)}, ...}
    where -1, 1 was the old action space bound.
    old action space is needed to rescale actions.
    """
    extended_args = deepcopy(args)
  
    for arg in args:
        if is_iterable_args(args[arg]):
            extend_nestable_args(space[arg], extended_args, arg, args[arg], fn)
        else:
            extended_args[arg] = fn(space[arg], args[arg], fn)
    return extended_args


@extend_nestable_args.register(Dict)
def _extend_nestable_dict_args(space: Space, extended_args: dict, arg: str, args, fn):
    extended_args = extended_args[arg]
    
    for arg in args:
        if is_iterable_args(args[arg]):
            extend_nestable_args(space[arg], extended_args, arg, args[arg], fn)   
        else:
            extended_args[arg] = fn(space[arg], args[arg], fn)


@extend_nestable_args.register(Tuple)
def _extend_nestable_tuple_args(space: Space, extended_args: dict, space_idx: int, args, fn):  
    extended_args = extended_args[space_idx]

    if args is None:
        return

    for i, arg in enumerate(args):
        if is_iterable_args(args[i]):
            extend_nestable_args(space[i], extended_args, i, args[i], fn)
        else:
            extended_args[i] = fn(space[i], arg, fn)
