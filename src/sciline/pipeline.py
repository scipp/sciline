# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from sciline.task_graph import TaskGraph

from .domain import Scope
from .scheduler import Graph, Scheduler

T = TypeVar('T')


class UnsatisfiedRequirement(Exception):
    """Raised when a type cannot be provided."""


class UnboundTypeVar(Exception):
    """
    Raised when a parameter of a generic provider is not bound to a concrete type.
    """


class AmbiguousProvider(Exception):
    """Raised when multiple providers are found for a type."""


Provider = Callable[..., Any]
Key = type


def _is_compatible_type_tuple(
    requested: tuple[Key, ...],
    provided: tuple[Key | TypeVar, ...],
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


def _bind_free_typevars(tp: TypeVar | Key, bound: Dict[TypeVar, Key]) -> Key:
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


class Pipeline:
    """A container for providers that can be assembled into a task graph."""

    def __init__(
        self,
        providers: Optional[List[Provider]] = None,
        *,
        params: Optional[Dict[Key, Any]] = None,
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
        self._providers: Dict[Key, Provider] = {}
        self._subproviders: Dict[type, Dict[Tuple[Key | TypeVar, ...], Provider]] = {}
        for provider in providers or []:
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

    def __setitem__(self, key: Type[T], param: T) -> None:
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
            underlying = key.__supertype__  # type: ignore[attr-defined]
        else:
            underlying = key
        if (origin := get_origin(underlying)) is None:
            # In Python 3.8, get_origin does not work with numpy.typing.NDArray,
            # but it defines __origin__
            if (np_origin := getattr(underlying, '__origin__', None)) is not None:
                expected = np_origin
            else:
                expected = underlying
        elif issubclass(origin, Scope):
            scope = origin.__orig_bases__[0]
            while (orig := get_origin(scope)) is not None and orig is not Scope:
                scope = orig.__orig_bases__[0]
            expected = get_args(scope)[1]
        else:
            expected = origin

        if not isinstance(param, expected):
            raise TypeError(
                f'Key {key} incompatible to value {param} of type {type(param)}'
            )
        self._set_provider(key, lambda: param)

    def _set_provider(self, key: Type[T], provider: Callable[..., T]) -> None:
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

    def _get_provider(self, tp: Type[T]) -> Tuple[Callable[..., T], Dict[TypeVar, Key]]:
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

    def build(self, tp: Type[T], /) -> Graph:
        """
        Return a dict of providers required for building the requested type `tp`.

        This is mainly for internal and low-level use. Prefer using :py:meth:`get`.

        The values are tuples containing the provider and the dict of arguments for
        the provider. The values in the latter dict reference other keys in the returned
        graph.

        Parameters
        ----------
        tp:
            Type to build the graph for.
        """
        graph: Graph = {}
        stack: List[Type[T]] = [tp]
        while stack:
            tp = stack.pop()
            provider: Callable[..., T]
            provider, bound = self._get_provider(tp)
            tps = get_type_hints(provider)
            args = {
                name: _bind_free_typevars(t, bound=bound)
                for name, t in tps.items()
                if name != 'return'
            }
            graph[tp] = (provider, args)
            for arg in args.values():
                if arg not in graph:
                    stack.append(arg)
        return graph

    @overload
    def compute(self, tp: Type[T]) -> T:
        ...

    @overload
    def compute(self, tp: Tuple[Type[T], ...]) -> Tuple[T, ...]:
        ...

    def compute(self, tp: type | Tuple[type, ...]) -> Any:
        """
        Compute result for the given keys.

        Equivalent to ``self.get(tp).compute()``.

        Parameters
        ----------
        tp:
            Type to compute the result for. Can be a single type or a tuple of types.
        """
        return self.get(tp).compute()

    def visualize(
        self, tp: type | Tuple[type, ...], **kwargs: Any
    ) -> graphviz.Digraph:  # type: ignore[name-defined] # noqa: F821
        """
        Return a graphviz Digraph object representing the graph for the given keys.

        Equivalent to ``self.get(tp).visualize()``.

        Parameters
        ----------
        tp:
            Type to visualize the graph for. Can be a single type or a tuple of types.
        kwargs:
            Keyword arguments passed to :py:class:`graphviz.Digraph`.
        """
        return self.get(tp).visualize(**kwargs)

    def get(
        self, keys: type | Tuple[type, ...], *, scheduler: Optional[Scheduler] = None
    ) -> TaskGraph:
        """
        Return a TaskGraph for the given keys.

        Parameters
        ----------
        keys:
            Type to compute the result for. Can be a single type or a tuple of types.
        scheduler:
            Optional scheduler to use for computing the result. If not given, a
            :py:class:`NaiveScheduler` is used if `dask` is not installed,
            otherwise dask's threaded scheduler is used.
        """
        if isinstance(keys, tuple):
            graph: Graph = {}
            for t in keys:
                graph.update(self.build(t))
        else:
            graph = self.build(keys)
        return TaskGraph(graph=graph, keys=keys, scheduler=scheduler)
