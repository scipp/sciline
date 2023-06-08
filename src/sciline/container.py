# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import typing
from typing import Callable, List, Type, TypeVar

import injector

T = TypeVar('T')


class Container:
    def __init__(self, inj: injector.Injector, /) -> None:
        self._injector = inj

    def get(self, tp: Type[T], /) -> T:
        return self._injector.get(tp)


def _injectable(func: Callable) -> Callable:
    """
    Wrap a regular function so it can be registered in an injector and have its
    parameters injected.
    """
    tps = typing.get_type_hints(func)

    def bind(binder: injector.Binder):
        binder.bind(tps['return'], injector.inject(func))

    return bind


def make_container(funcs: List[Callable], /) -> Container:
    """
    Create a :py:class:`Container` from a list of functions.

    Parameters
    ----------
    funcs:
        List of functions to be injected. Must be annotated with type hints.
    """
    return Container(injector.Injector([_injectable(func) for func in funcs]))
