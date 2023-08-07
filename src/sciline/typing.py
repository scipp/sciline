# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Tuple, Type, TypeVar, Union


@dataclass(frozen=True)
class Label:
    tp: type
    index: int


T = TypeVar('T')


@dataclass(frozen=True)
class Item(Generic[T]):
    label: Tuple[Label, ...]
    tp: Type[T]


Key = Union[type, Item]
Graph = Dict[
    Key,
    Tuple[Callable[..., Any], Tuple[Key, ...]],
]
