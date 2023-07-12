# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Type,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import dask
from dask.delayed import Delayed

T = TypeVar('T')


class UnsatisfiedRequirement(Exception):
    pass


class AmbiguousProvider(Exception):
    pass


def _is_compatible_type_tuple(
    requested: tuple[type, ...], provided: tuple[type | TypeVar, ...]
) -> bool:
    """
    Check if a tuple of requested types is compatible with a tuple of provided types.

    Types in the tuples must either by equal, or the provided type must be a TypeVar.
    """
    for req, prov in zip(requested, provided):
        if isinstance(prov, TypeVar):
            continue
        if req != prov:
            return False
    return True


Provider = Callable[..., Any]


class Container:
    def __init__(self, funcs: List[Provider], /):
        """
        Create a :py:class:`Container` from a list of functions.

        Parameters
        ----------
        funcs:
            List of functions to be injected. Must be annotated with type hints.
        """
        self._providers: Dict[type, Provider] = {}
        self._generic_providers: Dict[
            type, Dict[Tuple[type | TypeVar, ...], Provider]
        ] = {}
        self._cache: Dict[type, Delayed] = {}
        for func in funcs:
            self.insert(func)

    def insert(self, provider: Provider) -> None:
        if (key := get_type_hints(provider).get('return')) is None:
            raise ValueError(f'Provider {provider} lacks type-hint for return value')
        # isinstance does not work here and types.NoneType available only in 3.10+
        if key == type(None):  # noqa: E721
            raise ValueError(f'Provider {provider} returning `None` is not allowed')
        if get_origin(key) is not None:
            subproviders = self._generic_providers.setdefault(get_origin(key), {})
            args = get_args(key)
            if args in subproviders:
                raise ValueError(f'Provider for {key} already exists')
            subproviders[args] = provider
        else:
            if key in self._providers:
                raise ValueError(f'Provider for {key} already exists')
            self._providers[key] = provider

    def _call(self, func: Callable[..., Any], bound: Dict[TypeVar, type]) -> Delayed:
        tps = get_type_hints(func)
        del tps['return']
        args = {
            name: self._get(self._bind_free_typevars(tp, bound=bound))
            for name, tp in tps.items()
        }
        return dask.delayed(func)(**args)

    def _bind_free_typevars(self, tp: type, bound: Dict[TypeVar, type]) -> type:
        if isinstance(tp, TypeVar):
            return bound[tp]
        elif (origin := get_origin(tp)) is not None:
            return origin[tuple(bound.get(a, a) for a in get_args(tp))]
        else:
            return tp

    def _get(self, tp: Type[T], /) -> Delayed:
        # When building a workflow, there are two common problems:
        #
        # 1. Intermediate results are used more than once.
        # 2. Intermediate results are large, so we generally do not want to keep them
        #    in memory longer than necessary.
        #
        # To address these problems, we can internally build a graph of tasks, instead
        # of directly creating dependencies between functions. Currently we use Dask
        # for this. We cache call results to ensure Dask will recognize the task
        # as the same object) and also wrap the function in dask.delayed.
        if (cached := self._cache.get(tp)) is not None:
            return cached

        if (provider := self._providers.get(tp)) is not None:
            result = self._call(provider, {})
        elif (origin := get_origin(tp)) is not None and (
            subproviders := self._generic_providers[origin]
        ) is not None:
            requested = get_args(tp)
            matches = [
                (args, subprovider)
                for args, subprovider in subproviders.items()
                if _is_compatible_type_tuple(requested, args)
            ]
            if len(matches) == 0:
                raise UnsatisfiedRequirement("No provider found for type", tp)
            elif len(matches) > 1:
                raise AmbiguousProvider("Multiple providers found for type", tp)
            args, provider = matches[0]
            bound = {
                arg: req
                for arg, req in zip(args, requested)
                if isinstance(arg, TypeVar)
            }
            result = self._call(provider, bound)
        else:
            raise UnsatisfiedRequirement("No provider found for type", tp)

        self._cache[tp] = result
        return result

    def get(self, tp: Type[T], /) -> Delayed:
        # We are slightly abusing Python's type system here, by using the
        # self._get to get T, but actually it returns a Delayed that can
        # compute T. We'd like to use Delayed[T], but that is not supported yet:
        # https://github.com/dask/dask/pull/9256
        return self._get(tp)

    def compute(self, tp: Type[T], /) -> T:
        task = self.get(tp)
        result: T = task.compute()  # type: ignore
        return result
