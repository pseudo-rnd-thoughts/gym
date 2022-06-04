from typing import TypeVar, Union, Dict, Sequence

ArgType = TypeVar('ArgType')

# TODO:
# FuncArgType = Union[None, ArgType, Dict[str, "FuncArgType"], Sequence["FuncArgType", ...]]
FuncArgType = Union[None, ArgType, Dict[str, "FuncArgType"], Sequence["FuncArgType"]]
