"""Implementation of utility functions that can be applied to spaces.

These functions mostly take care of flattening and unflattening elements of spaces to facilitate their usage in learning code.
"""
from __future__ import annotations

import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Any, Callable, Optional, Sequence, TypeVar, Union, cast

import numpy as np

from gym.dev_wrappers import FuncArgType
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple


@singledispatch
def flatdim(space: Space) -> int:
    """Return the number of dimensions a flattened equivalent of this space would have.

    Accepts a space and returns an integer.

    Raises:
         NotImplementedError: if the space is not defined in ``gym.spaces``.

    Example usage::

        >>> s = Dict({"position": Discrete(2), "velocity": Discrete(3)})
        >>> flatdim(s)
        5
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@flatdim.register(Box)
@flatdim.register(MultiBinary)
def _flatdim_box_multibinary(space: Union[Box, MultiBinary]) -> int:
    return reduce(op.mul, space.shape, 1)


@flatdim.register(Discrete)
def _flatdim_discrete(space: Discrete) -> int:
    return int(space.n)


@flatdim.register(MultiDiscrete)
def _flatdim_multidiscrete(space: MultiDiscrete) -> int:
    return int(np.sum(space.nvec))


@flatdim.register(Tuple)
def _flatdim_tuple(space: Tuple) -> int:
    return sum(flatdim(s) for s in space.spaces)


@flatdim.register(Dict)
def _flatdim_dict(space: Dict) -> int:
    return sum(flatdim(s) for s in space.spaces.values())


T = TypeVar("T")


@singledispatch
def flatten(space: Space[T], x: T) -> np.ndarray:
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Accepts a space and a point from that space. Always returns a 1D array.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@flatten.register(Box)
@flatten.register(MultiBinary)
def _flatten_box_multibinary(space, x) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).flatten()


@flatten.register(Discrete)
def _flatten_discrete(space, x) -> np.ndarray:
    onehot = np.zeros(space.n, dtype=space.dtype)
    onehot[x - space.start] = 1
    return onehot


@flatten.register(MultiDiscrete)
def _flatten_multidiscrete(space, x) -> np.ndarray:
    offsets = np.zeros((space.nvec.size + 1,), dtype=space.dtype)
    offsets[1:] = np.cumsum(space.nvec.flatten())

    onehot = np.zeros((offsets[-1],), dtype=space.dtype)
    onehot[offsets[:-1] + x.flatten()] = 1
    return onehot


@flatten.register(Tuple)
def _flatten_tuple(space, x) -> np.ndarray:
    return np.concatenate([flatten(s, x_part) for x_part, s in zip(x, space.spaces)])


@flatten.register(Dict)
def _flatten_dict(space, x) -> np.ndarray:
    return np.concatenate(
        [flatten(s, x[key]) for key, s in space.spaces.items()]
    )  # TODO, why do we not return a dictionary here?


@singledispatch
def unflatten(space: Space[T], x: np.ndarray) -> T:
    """Unflatten a data point from a space.

    This reverses the transformation applied by :func:`flatten`. You must ensure
    that the ``space`` argument is the same as for the :func:`flatten` call.

    Accepts a space and a flattened point. Returns a point with a structure
    that matches the space. Raises ``NotImplementedError`` if the space is not
    defined in ``gym.spaces``.
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@unflatten.register(Box)
@unflatten.register(MultiBinary)
def _unflatten_box_multibinary(space: Box | MultiBinary, x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=space.dtype).reshape(space.shape)


@unflatten.register(Discrete)
def _unflatten_discrete(space: Discrete, x: np.ndarray) -> int:
    return int(space.start + np.nonzero(x)[0][0])


@unflatten.register(MultiDiscrete)
def _unflatten_multidiscrete(space: MultiDiscrete, x: np.ndarray) -> np.ndarray:
    offsets = np.zeros((space.nvec.size + 1,), dtype=space.dtype)
    offsets[1:] = np.cumsum(space.nvec.flatten())

    (indices,) = cast(type(offsets[:-1]), np.nonzero(x))
    return np.asarray(indices - offsets[:-1], dtype=space.dtype).reshape(space.shape)


@unflatten.register(Tuple)
def _unflatten_tuple(space: Tuple, x: np.ndarray) -> tuple:
    dims = np.asarray([flatdim(s) for s in space.spaces], dtype=np.int_)
    list_flattened = np.split(x, np.cumsum(dims[:-1]))
    return tuple(
        unflatten(s, flattened) for flattened, s in zip(list_flattened, space.spaces)
    )


@unflatten.register(Dict)
def _unflatten_dict(space: Dict, x: np.ndarray) -> dict:
    dims = np.asarray([flatdim(s) for s in space.spaces.values()], dtype=np.int_)
    list_flattened = np.split(x, np.cumsum(dims[:-1]))
    return OrderedDict(
        [
            (key, unflatten(s, flattened))
            for flattened, (key, s) in zip(list_flattened, space.spaces.items())
        ]
    )


@singledispatch
def flatten_space(space: Space) -> Box:
    """Flatten a space into a single ``Box``.

    This is equivalent to :func:`flatten`, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    :func:`flatdim` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.

    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.

    Example::

        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that flattens a discrete space::

        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that recursively flattens a dict::

        >>> space = Dict({"position": Discrete(2), "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    raise NotImplementedError(f"Unknown space: `{space}`")


@flatten_space.register(Box)
def _flatten_space_box(space: Box) -> Box:
    return Box(space.low.flatten(), space.high.flatten(), dtype=space.dtype)


@flatten_space.register(Discrete)
@flatten_space.register(MultiBinary)
@flatten_space.register(MultiDiscrete)
def _flatten_space_binary(space: Union[Discrete, MultiBinary, MultiDiscrete]) -> Box:
    return Box(low=0, high=1, shape=(flatdim(space),), dtype=space.dtype)


@flatten_space.register(Tuple)
def _flatten_space_tuple(space: Tuple) -> Box:
    space_list = [flatten_space(s) for s in space.spaces]
    return Box(
        low=np.concatenate([s.low for s in space_list]),
        high=np.concatenate([s.high for s in space_list]),
        dtype=np.result_type(*[s.dtype for s in space_list]),
    )


@flatten_space.register(Dict)
def _flatten_space_dict(space: Dict) -> Box:
    space_list = [flatten_space(s) for s in space.spaces.values()]
    return Box(
        low=np.concatenate([s.low for s in space_list]),
        high=np.concatenate([s.high for s in space_list]),
        dtype=np.result_type(*[s.dtype for s in space_list]),
    )


@singledispatch
def apply_function(space: Space, x, func: Callable, args: FuncArgType[Any]) -> Any:
    """Applies a function on ``x`` of shape ``space`` using the ``func`` callable and ``args`` arguments.

    Example with fundamental space::
        TODO

    Example with dict (composite) space::
        TODO

    Example with tuple (composite) space::
        TODO

    Args:
        space: The space of ``x``
        x: The parameter to apply the function to
        func: The function to apply to ``x``
        args: The arguments to use with the function

    Returns:
        The updated ``x`` through the applied function and arguments
    """


@apply_function.register(Box)
@apply_function.register(Discrete)
@apply_function.register(MultiDiscrete)
@apply_function.register(MultiBinary)
def _apply_function_fundamental(_, x: Any, func: Callable, *args: Optional[Any]):
    return func(x, *args)


@apply_function.register(Dict)
def _apply_function_dict(space: Dict, x: Any, func: Callable, args: Optional[Any]):
    def _apply_function_dict_helper(
        updated_x: Any, space: Space, x: Any, space_key: str, func: Callable, args: Dict
    ):
        if space_key not in args:
            updated_x[space_key] = x.get(space_key)
            return updated_x

        space = space[space_key]
        args = args[space_key]
        x = x[space_key]

        if type(space) != Dict:
            updated_x[space_key] = apply_function(space, x, func, args)
        else:
            updated_x[space_key] = OrderedDict()
            for nested_space_key in space:
                _apply_function_dict_helper(
                    updated_x[space_key], space, x, nested_space_key, func, args
                )

        return updated_x

    if args is None:
        return OrderedDict(
            [
                (space_key, apply_function(subspace, x[space_key], func, None))
                for space_key, subspace in space.spaces.items()
            ]
        )

    elif isinstance(args, dict):
        updated_x = OrderedDict()
        for k in space:
            updated_x = _apply_function_dict_helper(updated_x, space, x, k, func, args)
        return updated_x

    else:
        raise Exception  # TODO, maybe unsure


@apply_function.register(Tuple)
def _apply_function_tuple(space: Tuple, x: Any, func: Callable, args: Optional[Any]):
    if args is None:
        return tuple(
            apply_function(subspace, val, func, None)
            for subspace, val in zip(space.spaces, x)
        )
    elif isinstance(args, Sequence):
        assert len(args) == len(
            space
        )  # TODO, not sure if we can deal with less args than
        return tuple(
            apply_function(subspace, val, func, arg)
            for subspace, val, arg in zip(space.spaces, x, args)
        )
    else:
        raise Exception  # TODO
