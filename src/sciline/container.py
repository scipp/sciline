# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import typing
from functools import wraps
from typing import Callable, List, Type, TypeVar, Union

import injector
from dask.delayed import Delayed

T = TypeVar('T')


class UnsatisfiedRequirement(Exception):
    pass


class Container:
    def __init__(self, inj: injector.Injector, /, *, lazy: bool) -> None:
        self._injector = inj
        self._lazy = lazy

    def get(self, tp: Type[T], /) -> Union[T, Delayed]:
        try:
            # We are slightly abusing Python's type system here, by using the
            # injector to get T, but actually it returns a Delayed that can
            # compute T. self._injector does not know this due to how we setup the
            # bindings. We'd like to use Delayed[T], but that is not supported yet:
            # https://github.com/dask/dask/pull/9256
            task: Delayed = self._injector.get(tp)  # type: ignore
        except injector.UnsatisfiedRequirement as e:
            raise UnsatisfiedRequirement(e) from e
        return task if self._lazy else task.compute()

    def make_child_container(self, funcs: List[Callable], /) -> Container:
        """
        Create a child container from a list of functions.

        The child container inherits all bindings from the parent container, but
        can override them with new bindings.

        Warning
        -------

        Note that it is not possible to override transitive dependencies, i.e., if the
        parent container provides A, and A depends on B, then the child container
        cannot override the B that is used by A. It can only override the B that is
        used by the child container.

        Parameters
        ----------
        funcs:
            List of functions to be injected. Must be annotated with type hints.
        """
        return Container(
            self._injector.create_child_injector([_injectable(f) for f in funcs]),
            lazy=self._lazy,
        )


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


def _injectable(func: Callable) -> Callable:
    """
    Wrap a regular function so it can be registered in an injector and have its
    parameters injected.
    """
    # When building a workflow, there are two common problems:
    #
    # 1. Intermediate results are used more than once.
    # 2. Intermediate results are large, so we generally do not want to keep them
    #    in memory longer than necessary.
    #
    # To address these problems, we can internally build a graph of tasks, instead of
    # directly creating dependencies between functions. Currently we use Dask for this.
    # The Container instance will automatically compute the task, unless it is marked
    # as lazy. We therefore use singleton-scope (to ensure Dask will recognize the
    # task as the same object) and also wrap the function in dask.delayed.
    scope = injector.singleton
    func = _delayed(func)
    tps = typing.get_type_hints(func)

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
        injector.Injector([_injectable(f) for f in funcs], auto_bind=False),
        lazy=lazy,
    )
