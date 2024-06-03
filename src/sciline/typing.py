# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, TypeVar, Union, get_args, get_origin

from ._provider import Provider

T = TypeVar('T')


Key = type
Graph = dict[Key, Provider]


Json = dict[str, "Json"] | list["Json"] | str | int | float | bool | None


def get_optional(tp: Key) -> Any | None:
    if get_origin(tp) != Union:
        return None
    args = get_args(tp)
    if len(args) != 2 or type(None) not in args:
        return None
    return args[0] if args[1] == type(None) else args[1]  # noqa: E721


def get_union(tp: Key) -> tuple[Any, ...] | None:
    if get_origin(tp) != Union:
        return None
    return get_args(tp)
