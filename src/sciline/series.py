# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections import abc
from typing import Generic, Iterator, Mapping, Type, TypeVar

Key = TypeVar('Key')
Value = TypeVar('Value')


class Series(abc.Mapping, Generic[Key, Value]):
    def __init__(self, row_dim: Type[Key], values: Mapping[Key, Value]) -> None:
        self._row_dim = row_dim
        self._map: Mapping[Key, Value] = values

    @property
    def row_dim(self) -> type:
        return self._row_dim

    def __contains__(self, item: object) -> bool:
        return item in self._map

    def __iter__(self) -> Iterator[Key]:
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def __getitem__(self, key: Key) -> Value:
        return self._map[key]

    def __repr__(self) -> str:
        return f"Series(row_dim={self.row_dim}, {self._map})"

    def _repr_html_(self) -> str:
        return (
            f"<table><tr><th>{self.row_dim.__name__}</th><th>Value</th></tr>"
            + "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in self._map.items()
            )
            + "</table>"
        )