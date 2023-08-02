# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections import abc
from typing import Generic, Iterator, Mapping, TypeVar

Key = TypeVar('Key')
Value = TypeVar('Value')


class Series(abc.Mapping, Generic[Key, Value]):
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
