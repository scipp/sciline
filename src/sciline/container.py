# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import typing
from typing import Callable, List, Type, TypeVar

import injector

T = TypeVar('T')


def injectable(func: Callable):
    tps = typing.get_type_hints(func)

    def bind(binder: injector.Binder):
        binder.bind(tps['return'], injector.inject(func))

    return bind


class Container:
    def __init__(self, inj: injector.Injector) -> None:
        self._injector = inj

    def get(self, tp: Type[T]) -> T:
        return self._injector.get(tp)


def make_container(funcs: List[Callable]) -> Container:
    return Container(injector.Injector([injectable(func) for func in funcs]))
