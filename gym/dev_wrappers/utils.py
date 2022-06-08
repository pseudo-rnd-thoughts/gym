"""A set of utility functions for lambda wrappers."""
from gym import Space


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