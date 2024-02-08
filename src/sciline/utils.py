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


def qualname(obj: Any) -> str:
    try:
        obj_name = obj.__qualname__
    except AttributeError:
        obj_name = str(obj)

    module = inspect.getmodule(obj)
    if module is None or module.__name__ == 'builtins':
        return obj_name
    return f'{module.__name__}.{obj_name}'


def keyname(key: Union[Key, TypeVar]) -> str:
    if isinstance(key, TypeVar):
        return str(key)
    if isinstance(key, Item):
        return (
            f'{keyname(key.tp)}({", ".join(keyname(label.tp) for label in key.label)})'
        )
    args = get_args(key)
    if len(args):
        parameters = ', '.join(map(keyname, args))
        return f'{key.__name__}[{parameters}]'
    return key.__name__


def keyqualname(key: Union[Key, TypeVar]) -> str:
    if isinstance(key, TypeVar):
        return qualname(key)
    if isinstance(key, Item):
        return (
            f'{keyqualname(key.tp)}'
            f'({", ".join(keyqualname(label.tp) for label in key.label)})'
        )
    args = get_args(key)
    if len(args):
        return str(key)
    return qualname(key)
