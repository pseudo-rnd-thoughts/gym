from typing import TypeVar, Union, Dict, Sequence

ArgType = TypeVar('ArgType')
FuncArgType = Union[None, ArgType, Dict[str, "FuncArgType"], Sequence["FuncArgType", ...]]
