"""A set of utility functions for lambda wrappers."""
from copy import deepcopy
from functools import singledispatch
from typing import Any, Callable, Optional, Sequence
from typing import Tuple as TypingTuple

import gym
from gym import Space
from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple


def extend_args(action_space: Space, extended_args: dict, args: dict, space_key: str):
    """Extend args for rescaling actions.

    Action space args needs to be extended in order
    to correctly rescale the actions.
    i.e. args before: {"body":{"left_arm": (-0.5,0.5)}, ...}
    args after: {"body":{"left_arm": (-0.5,0.5,-1,1)}, ...}
    where -1, 1 was the old action space bound.
    old action space is needed to rescale actions.
    """
    if space_key not in args:
        return extended_args

    args = args[space_key]

    if isinstance(args, dict):
        extended_args[space_key] = {}
        for arg in args:
            extend_args(action_space[space_key], extended_args[space_key], args, arg)
    else:
        assert len(args) == len(action_space[space_key].low) + len(
            action_space[space_key].high
        )
        extended_args[space_key] = (
            *args,
            *list(action_space[space_key].low),
            *list(action_space[space_key].high),
        )

    return extended_args


def is_nestable(space: Space):
    """Returns whether the input space can contains other spaces."""
    return isinstance(space, Tuple) or isinstance(space, Dict)


@singledispatch
def transform_space(
    space: Space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
) -> Any:
    """Transform space with the provided args."""


@transform_space.register(Box)
def _transform_space_box(space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]):
    if not args:
        return space
    return Box(*args, shape=space.shape)


@transform_space.register(Discrete)
def _transform_space_discrete(
    space, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
):
    if not args:
        return space
    return space


@transform_space.register(Tuple)
def _transform_space_tuple(
    space: Tuple, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
):
    assert isinstance(args, Sequence)
    assert len(space) == len(args)

    updated_space = [s for s in space]

    for i, arg in enumerate(args):
        if is_nestable(space[i]):
            transform_nestable_space(space[i], updated_space, i, args[i], env)
            if isinstance(updated_space[i], list):
                updated_space[i] = Tuple(updated_space[i])
        else:
            updated_space[i] = transform_space(env.action_space[i], env, arg)

    return Tuple(updated_space)


@transform_space.register(Dict)
def _transform_space_dict(
    space: Dict, env: gym.Env, args: FuncArgType[TypingTuple[int, int]]
):
    assert isinstance(args, dict)
    updated_space = deepcopy(env.action_space)

    for k in args:
        if is_nestable(space[k]):
            transform_nestable_space(space[k], updated_space, k, args[k], env)
        else:
            updated_space[k] = transform_space(space[k], env, args.get(k))
    return updated_space


@singledispatch
def transform_nestable_space(
    original_space: gym.Space,
    space: gym.Space,
    space_key: str,
    args: FuncArgType[TypingTuple[int, int]],
    env=None,
):
    """Transform nestable space with the provided args."""


@transform_nestable_space.register(Dict)
def _transform_nestable_dict_space(
    original_space: gym.Space,
    updated_space: gym.Space,
    key_to_update: str,
    args: FuncArgType[TypingTuple[int, int]],
    env,
):
    """Recursive function to process possibly nested `Dict` spaces."""
    updated_space = updated_space[key_to_update]

    for k in args:
        if is_nestable(original_space[k]):
            transform_nestable_space(original_space[k], updated_space, k, args[k], env)
        else:
            updated_space[k] = transform_space(original_space[k], env, args.get(k))


@transform_nestable_space.register(Tuple)
def _transform_nestable_tuple_space(
    original_space: gym.Space,
    updated_space: gym.Space,
    idx_to_update: int,
    args: FuncArgType[TypingTuple[int, int]],
    env,
):
    """Recursive function to process possibly nested `Tuple` spaces."""
    updated_space[idx_to_update] = [s for s in original_space]
    updated_space = updated_space[idx_to_update]

    if args is None:
        return

    for i, arg in enumerate(args):
        if is_nestable(original_space[i]):
            transform_nestable_space(original_space[i], updated_space, i, args[i], env)
        else:
            updated_space[i] = transform_space(original_space[i], env, arg)
