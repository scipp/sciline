# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import inspect
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Iterable, TypeVar, Union, get_args

from .typing import Item, Key

T = TypeVar('T')
G = TypeVar('G')


def groupby(f: Callable[[T], G], a: Iterable[T]) -> DefaultDict[G, list[T]]:
    g = defaultdict(lambda: [])
    for e in a:
        g[f(e)].append(e)
    return g


def fullqualname(obj: Any) -> str:
    try:
        obj_name = obj.__qualname__
    except AttributeError:
        obj_name = str(obj)

    module = inspect.getmodule(obj)
    if module is None:
        return obj_name
    return f'{module.__name__}.{obj_name}'


def keyname(key: Union[Key, TypeVar]) -> str:
    if isinstance(key, Item):
        return (
            f'{keyname(key.tp)}({", ".join(keyname(label.tp) for label in key.label)})'
        )
    args = get_args(key)
    if len(args):
        parameters = ', '.join(map(keyname, args))
        return f'{key.__name__}[{parameters}]'
    return key.__name__


def keyfullqualname(key: Union[Key, TypeVar]) -> str:
    if isinstance(key, Item):
        return (
            f'{keyfullqualname(key.tp)}'
            f'({", ".join(keyfullqualname(label.tp) for label in key.label)})'
        )
    args = get_args(key)
    if len(args):
        parameters = ', '.join(map(keyfullqualname, args))
        return f'{fullqualname(key.__origin__)}[{parameters}]'
    return fullqualname(key)
