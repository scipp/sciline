# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import typing
from functools import wraps
from typing import Callable, List, Type, TypeVar

import injector

T = TypeVar('T')


class UnsatisfiedRequirement(Exception):
    pass


class Container:
    def __init__(self, inj: injector.Injector, /) -> None:
        self._injector = inj

    def get(self, tp: Type[T], /) -> T:
        try:
            return self._injector.get(tp)
        except injector.UnsatisfiedRequirement as e:
            raise UnsatisfiedRequirement(e) from e


def _delayed(func: Callable) -> Callable:
    """
    Decorator to make a function return a delayed object.

    In contrast to dask.delayed, this uses functools.wraps, to preserve the
    type hints, which is a prerequisite for injector to work.
    """
    import dask

    @wraps(func)
    def wrapper(*args, **kwargs):
        return dask.delayed(func)(*args, **kwargs)

    return wrapper


def _injectable(func: Callable, *, lazy: bool) -> Callable:
    """
    Wrap a regular function so it can be registered in an injector and have its
    parameters injected.
    """
    tps = typing.get_type_hints(func)
    # When lazy, we want to create a dask task graph without duplicate computation.
    # This means we need to use a singleton scope, so that the function is only
    # called once.
    # When not lazy, we have to avoid memory consumption from injector holding on
    # to large intermediate results. We therefore do not use a singleton scope.
    # This is however problematic as multiple functions may rely on a result
    # from a previous function. Maybe we need more manual control? Or maybe we can
    # always use dask, but compute() automatically when not lazy?
    scope = injector.singleton if lazy else None
    if lazy:
        func = _delayed(func)

    def bind(binder: injector.Binder):
        binder.bind(tps['return'], injector.inject(func), scope=scope)

    return bind


def make_container(funcs: List[Callable], /, *, lazy: bool = False) -> Container:
    """
    Create a :py:class:`Container` from a list of functions.

    Parameters
    ----------
    funcs:
        List of functions to be injected. Must be annotated with type hints.
    lazy:
        If True, the functions are wrapped in :py:func:`dask.delayed` before
        being injected. This allows to build a dask graph from the container.
    """
    # Note that we disable auto_bind, to ensure we do not accidentally bind to
    # some default values. Everything must be explicit.
    return Container(
        injector.Injector(
            [_injectable(func, lazy=lazy) for func in funcs], auto_bind=False
        )
    )
