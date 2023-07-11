# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import typing
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import injector
from dask.delayed import Delayed


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


T = TypeVar('T')


class UnsatisfiedRequirement(Exception):
    pass


class Container2:
    def __init__(self):
        self._providers: Dict[type, Callable[..., Any]] = {}
        self._lazy: bool = False

    def insert(self, provider: Callable[..., Any]):
        key = get_type_hints(provider)['return']
        if (origin := get_origin(key)) is not None:
            args = get_args(key)
            if len(args) != 1:
                raise ValueError(f'Cannot handle {key} with more than 1 argument')
            key = origin if isinstance(args[0], TypeVar) else key
        if key in self._providers:
            raise ValueError(f'Provider for {key} already exists')
        self._providers[key] = _delayed(provider)

    Return = typing.TypeVar("Return")

    def call(self, func: Callable[..., Return], bound: Optional[Any] = None) -> Return:
        print('call', func, bound)
        tps = get_type_hints(func)
        del tps['return']
        args: Dict[str, Any] = {}
        for name, tp in tps.items():
            args[name] = self._get(tp, bound=bound)
        return func(**args)

    def _get(self, tp, bound: Optional[type] = None):
        if (provider := self._providers.get(tp)) is not None:
            return self.call(provider, bound)
        elif (origin := get_origin(tp)) is not None:
            if (provider := self._providers.get(origin)) is not None:
                # TODO We would really need to support multiple bound params properly
                param = get_args(tp)[0]
                return self.call(
                    provider, bound if isinstance(param, TypeVar) else param
                )
            else:
                provider = self._providers[origin[bound]]
                return self.call(provider, bound)
        raise UnsatisfiedRequirement("No provider found for type", tp)

    def get(self, tp: Type[T], /) -> Union[T, Delayed]:
        try:
            # We are slightly abusing Python's type system here, by using the
            # injector to get T, but actually it returns a Delayed that can
            # compute T. self._injector does not know this due to how we setup the
            # bindings. We'd like to use Delayed[T], but that is not supported yet:
            # https://github.com/dask/dask/pull/9256
            task: Delayed = self._get(tp)  # type: ignore
        except injector.UnsatisfiedRequirement as e:
            raise UnsatisfiedRequirement(e) from e
        return task if self._lazy else task.compute()


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
