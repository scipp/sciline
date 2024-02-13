# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import inspect
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Iterable, TypeVar, Union, get_args

from ._provider import Provider
from .typing import Item, Key

T = TypeVar('T')
G = TypeVar('G')


def groupby(f: Callable[[T], G], a: Iterable[T]) -> DefaultDict[G, list[T]]:
    g = defaultdict(lambda: [])
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


def key_name(key: Union[Key, TypeVar]) -> str:
    if isinstance(key, Item):
        parameters = ", ".join(
            f'{key_name(label.tp)}:{label.index}' for label in key.label
        )
        return f'{key_name(key.tp)}({parameters})'
    args = get_args(key)
    if len(args):
        parameters = ', '.join(map(key_name, args))
        # getattr is a fallback for python < 3.10
        return f'{getattr(key, "__name__", "<Generic>")}[{parameters}]'
    if isinstance(key, TypeVar):
        return str(key)
    return key.__name__


def key_full_qualname(key: Union[Key, TypeVar]) -> str:
    if isinstance(key, Item):
        parameters = ", ".join(
            f'{key_full_qualname(label.tp)}:{label.index}' for label in key.label
        )
        return f'{key_full_qualname(key.tp)}({parameters})'
    args = get_args(key)
    if len(args):
        origin = key.__origin__  # type: ignore[union-attr] # key is a TypeVar
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
