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


class Container:
    def __init__(self, funcs: List[Callable], /, *, lazy: bool = False):
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
        self._providers: Dict[type, Callable[..., Any]] = {}
        self._lazy: bool = lazy
        self._cache: Dict[type, Any] = {}
        for func in funcs:
            self.insert(func)

    def insert(self, provider: Callable[..., Any]) -> None:
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
        tps = get_type_hints(func)
        del tps['return']
        args: Dict[str, Any] = {}
        for name, tp in tps.items():
            if isinstance(tp, TypeVar):
                tp = tp if bound is None else bound
            elif (origin := get_origin(tp)) is not None:
                if isinstance(get_args(tp)[0], TypeVar):
                    tp = origin[bound]
            args[name] = self._get(tp)
        return func(**args)

    def _get(self, tp: Type[T], /) -> Delayed:
        # When building a workflow, there are two common problems:
        #
        # 1. Intermediate results are used more than once.
        # 2. Intermediate results are large, so we generally do not want to keep them
        #    in memory longer than necessary.
        #
        # To address these problems, we can internally build a graph of tasks, instead
        # of # directly creating dependencies between functions. Currently we use Dask
        # for this.  # The Container instance will automatically compute the task,
        # unless it is marked as lazy. We therefore use singleton-scope (to ensure Dask
        # will recognize the task as the same object) and also wrap the function in
        # dask.delayed.
        if tp in self._providers:
            key = tp
            bound = None
        elif (origin := get_origin(tp)) in self._providers:
            key = origin
            bound = get_args(tp)[0]
        else:
            raise UnsatisfiedRequirement("No provider found for type", tp)
        if (cached := self._cache.get(key)) is not None:
            return cached
        provider = self._providers.get(key)
        result = self.call(provider, bound)
        self._cache[tp] = result
        return result

    def get(self, tp: Type[T], /) -> Union[T, Delayed]:
        # We are slightly abusing Python's type system here, by using the
        # injector to get T, but actually it returns a Delayed that can
        # compute T. self._injector does not know this due to how we setup the
        # bindings. We'd like to use Delayed[T], but that is not supported yet:
        # https://github.com/dask/dask/pull/9256
        task: Delayed = self._get(tp)  # type: ignore
        return task if self._lazy else task.compute()
