# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)


@dataclass(frozen=True)
class Label:
    tp: type
    index: Any


T = TypeVar('T')


@dataclass(frozen=True)
class Item(Generic[T]):
    label: Tuple[Label, ...]
    tp: Type[T]


Provider = Callable[..., Any]


Key = Union[type, Item]
Graph = Dict[Key, Tuple[Provider, Tuple[Key, ...]]]


def get_optional(tp: Key) -> Optional[Any]:
    if get_origin(tp) != Union:
        return None
    args = get_args(tp)
    if len(args) != 2 or type(None) not in args:
        return None
    return args[0] if args[1] == type(None) else args[1]  # noqa: E721
