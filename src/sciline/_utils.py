# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import inspect
from collections import defaultdict
from collections.abc import Callable, Iterable
from types import UnionType
from typing import Any, TypeVar, get_args

from ._provider import Provider

T = TypeVar('T')
G = TypeVar('G')


def groupby(f: Callable[[T], G], a: Iterable[T]) -> defaultdict[G, list[T]]:
    g = defaultdict(list)
    for e in a:
        g[f(e)].append(e)
    return g


def full_qualname(obj: Any) -> str:
    try:
        obj_name = obj.__qualname__
    except AttributeError:
        obj_name = str(obj)

    module = inspect.getmodule(obj)
    if module is None:
        return str(obj_name)
    return f'{module.__name__}.{obj_name}'


def key_name(key: Any) -> str:
    args = get_args(key)
    if isinstance(key, UnionType):
        return ' | '.join(map(key_name, args))
    if len(args):
        parameters = ', '.join(map(key_name, args))
        return f'{key.__name__}[{parameters}]'
    if isinstance(key, TypeVar):
        return str(key)
    if hasattr(key, '__name__'):
        return str(key.__name__)
    return str(key)


def key_full_qualname(key: Any) -> str:
    args = get_args(key)
    if isinstance(key, UnionType):
        return ' | '.join(map(key_full_qualname, args))
    if len(args) and (origin := getattr(key, '__origin__', None)) is not None:
        # key is a type var
        parameters = ', '.join(map(key_full_qualname, args))
        return f'{full_qualname(origin)}[{parameters}]'
    return full_qualname(key)


def provider_name(provider: Provider) -> str:
    if provider.kind == 'function':
        return provider.func.__name__
    if provider.kind == 'parameter':
        return f'parameter({key_name(type(provider.call({})))})'
    return str(provider)


def provider_full_qualname(provider: Provider) -> str:
    if provider.kind == 'function':
        return full_qualname(provider.func)
    if provider.kind == 'parameter':
        return f'parameter({key_full_qualname(type(provider.call({})))})'
    return str(provider)
