# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from graphlib import TopologicalSorter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NewType,
    Optional,
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


class UnboundTypeVar(Exception):
    pass


class AmbiguousProvider(Exception):
    pass


def _is_compatible_type_tuple(
    requested: tuple[type | NewType, ...],
    provided: tuple[type | TypeVar | NewType, ...],
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


def _bind_free_typevars(tp: TypeVar | type, bound: Dict[TypeVar, type]) -> type:
    if isinstance(tp, TypeVar):
        if (result := bound.get(tp)) is None:
            raise UnboundTypeVar(f'Unbound type variable {tp}')
        return result
    elif (origin := get_origin(tp)) is not None:
        result = origin[tuple(_bind_free_typevars(arg, bound) for arg in get_args(tp))]
        if result is None:
            raise ValueError(f'Binding type variables in {tp} resulted in `None`')
        return result
    else:
        return tp


Provider = Callable[..., Any]
Key = type | NewType
Graph = Dict[
    Key,
    Tuple[Callable[..., Any], Dict[TypeVar, type], Dict[str, Key]],
]


class Pipeline:
    def __init__(
        self,
        providers: Optional[List[Provider]] = None,
        *,
        params: Optional[Dict[type | NewType, Any]] = None,
    ):
        """
        Setup a Pipeline from a list providers

        Parameters
        ----------
        providers:
            List of callable providers. Each provides its return value.
            Their arguments and return value must be annotated with type hints.
        params:
            Dictionary of concrete values to provide for types.
        """
        self._providers: Dict[type | NewType, Provider] = {}
        self._subproviders: Dict[
            type, Dict[Tuple[type | TypeVar | NewType, ...], Provider]
        ] = {}
        self._cache: Dict[type | NewType, Delayed] = {}
        providers = providers or []
        for provider in providers:
            self.insert(provider)
        for tp, param in (params or {}).items():
            self[tp] = param

    def insert(self, provider: Provider, /) -> None:
        """
        Add a callable that provides its return value to the pipeline.

        Parameters
        ----------
        provider:
            Callable that provides its return value. Its arguments and return value
            must be annotated with type hints.
        """
        if (key := get_type_hints(provider).get('return')) is None:
            raise ValueError(f'Provider {provider} lacks type-hint for return value')
        self._set_provider(key, provider)

    def __setitem__(self, key: Type[T] | NewType, param: T) -> None:
        """
        Provide a concrete value for a type.

        Parameters
        ----------
        key:
            Type to provide a value for.
        param:
            Concrete value to provide.
        """
        # TODO Switch to isinstance(key, NewType) once our minimum is Python 3.10
        # Note that we cannot pass mypy in Python<3.10 since NewType is not a type.
        if hasattr(key, '__supertype__'):
            expected = key.__supertype__
        elif (origin := get_origin(key)) is None:
            expected = key
        else:
            expected = origin
        if not isinstance(param, expected):
            raise TypeError(
                f'Key {key} incompatible to value {param} of type {type(param)}'
            )
        self._set_provider(key, lambda: param)

    def _set_provider(self, key: Type[T] | NewType, provider: Callable[..., T]) -> None:
        # isinstance does not work here and types.NoneType available only in 3.10+
        if key == type(None):  # noqa: E721
            raise ValueError(f'Provider {provider} returning `None` is not allowed')
        if (origin := get_origin(key)) is not None:
            subproviders = self._subproviders.setdefault(origin, {})
            args = get_args(key)
            if args in subproviders:
                raise ValueError(f'Provider for {key} already exists')
            subproviders[args] = provider
        else:
            if key in self._providers:
                raise ValueError(f'Provider for {key} already exists')
            self._providers[key] = provider

    def _get_args(
        self, func: Callable[..., Any], bound: Dict[TypeVar, type]
    ) -> Dict[str, Delayed]:
        return {
            name: self.get(_bind_free_typevars(tp, bound=bound))
            for name, tp in get_type_hints(func).items()
            if name != 'return'
        }

    def _get_provider(
        self, tp: Type[T]
    ) -> Tuple[Callable[..., T], Dict[TypeVar, type]]:
        if (provider := self._providers.get(tp)) is not None:
            return provider, {}
        elif (origin := get_origin(tp)) is not None and (
            subproviders := self._subproviders.get(origin)
        ) is not None:
            requested = get_args(tp)
            matches = [
                (args, subprovider)
                for args, subprovider in subproviders.items()
                if _is_compatible_type_tuple(requested, args)
            ]
            if len(matches) == 1:
                args, provider = matches[0]
                bound = {
                    arg: req
                    for arg, req in zip(args, requested)
                    if isinstance(arg, TypeVar)
                }
                return provider, bound
            elif len(matches) > 1:
                raise AmbiguousProvider("Multiple providers found for type", tp)
        raise UnsatisfiedRequirement("No provider found for type", tp)

    def build_graph(self, tp: Type[T], /) -> Graph:
        """
        Return a dict of providers required for building the provided type `tp`.

        The values are tuples container the provider, the dict of bound typevars,
        and the dict of arguments for the provider. The values in the latter dict
        reference other keys in the returned graph.

        Parameters
        ----------
        tp:
            Type to build the graph for.
        """
        provider, bound = self._get_provider(tp)
        tps = get_type_hints(provider)
        args = {
            name: _bind_free_typevars(t, bound=bound)
            for name, t in tps.items()
            if name != 'return'
        }
        graph = {tp: (provider, bound, args)}
        for arg in args.values():
            graph.update(self.build_graph(arg))
        return graph

    def get(self, tp: Type[T], /) -> Delayed:
        # We are slightly abusing Python's type system here, by using the
        # self.get to get T, but actually it returns a Delayed that can
        # compute T. We'd like to use Delayed[T], but that is not supported yet:
        # https://github.com/dask/dask/pull/9256
        #
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

        graph = self.build_graph(tp)
        dependencies = {tp: set(args.values()) for tp, (_, _, args) in graph.items()}
        ts = TopologicalSorter(dependencies)
        for key in ts.static_order():
            provider, _, args = graph[key]
            delayed = dask.delayed(provider)
            self._cache.setdefault(
                key, delayed(**{name: self._cache[arg] for name, arg in args.items()})
            )

        return self._cache[tp]

    def compute(self, tp: Type[T], /) -> T:
        task = self.get(tp)
        result: T = task.compute()  # type: ignore[no-untyped-call]
        return result


def as_dask_graph(graph):
    # Note: Only works if all providers support posargs
    return {tp: (provider, *args.values()) for tp, (provider, _, args) in graph.items()}
