# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import TypeVar, Iterator, Generic, List

from collections.abc import Collection, Mapping

T = TypeVar('T')


class Stack(Collection, Generic[T]):
    def __init__(self, values: List[T]) -> None:
        self._stack: List[T] = values

    def __contains__(self, item: object) -> bool:
        return item in self._stack

    def __iter__(self) -> Iterator[T]:
        return iter(self._stack)

    def __len__(self) -> int:
        return len(self._stack)


Key = TypeVar('Key')
Value = TypeVar('Value')


class Map(Mapping[Key, Value]):
    def __init__(self, values: Mapping[Key, Value]) -> None:
        self._map: Mapping[Key, Value] = values

    def __contains__(self, item: object) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[Key]:
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, key: Key) -> Value:
        return self._map[key]
