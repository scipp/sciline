# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from sciline.task_graph import TaskGraph

from .domain import Scope
from .graph import find_nodes_in_paths
from .scheduler import Graph, Scheduler
from .variadic import Map

T = TypeVar('T')


class UnsatisfiedRequirement(Exception):
    """Raised when a type cannot be provided."""


class UnboundTypeVar(Exception):
    """
    Raised when a parameter of a generic provider is not bound to a concrete type.
    """


class AmbiguousProvider(Exception):
    """Raised when multiple providers are found for a type."""


@dataclass(frozen=True)
class Label(Generic[T]):
    tp: Type[T]
    index: int


@dataclass(frozen=True)
class Item(Generic[T]):
    label: Tuple[Label[T], ...]
    tp: type


def _indexed_key(index_name: Any, i: int, value_name: Any) -> Union[Label, Item]:
    if index_name == value_name:
        return Label(index_name, i)
    label = Label(index_name, i)
    if isinstance(value_name, Item):
        return Item(value_name.label + (label,), value_name.tp)
    else:
        return Item((label,), value_name)


Provider = Callable[..., Any]
Key = Union[type, Label, Item]


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
        self._indices: Dict[Key, Collection[Any]] = {}
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

    def __setitem__(self, key: Union[Type[T], Label[T]], param: T) -> None:
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

    def set_index(self, key: Type[T], index: Collection[T]) -> None:
        self._indices[key] = index
        for i, label in enumerate(index):
            self._set_provider(Label(tp=key, index=i), lambda label=label: label)

    def _set_provider(
        self, key: Union[Type[T], Label[T]], provider: Callable[..., T]
    ) -> None:
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

    def _get_provider(
        self, tp: Union[Type[T], Label[T], Item]
    ) -> Tuple[Callable[..., T], Dict[TypeVar, Key]]:
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
        stack: List[Union[Type[T], Label[T]]] = [tp]
        while stack:
            tp = stack.pop()
            if tp in self._indices:
                continue
            if get_origin(tp) == Map:
                graph.update(self._build_indexed_subgraph(tp))
                continue
            provider: Callable[..., T]
            provider, bound = self._get_provider(tp)
            tps = get_type_hints(provider)
            args = tuple(
                _bind_free_typevars(t, bound=bound)
                for name, t in tps.items()
                if name != 'return'
            )
            graph[tp] = (provider, args)
            for arg in args:
                if arg not in graph:
                    stack.append(arg)
        return graph

    def _build_indexed_subgraph(self, tp: Type[T]) -> Graph:
        index_name, value_type = get_args(tp)
        size = len(self._indices[index_name])
        provider = self._make_mapping_provider(
            value_type=value_type, mapping_type=tp, index_name=index_name
        )
        args = [_indexed_key(index_name, i, value_type) for i in range(size)]
        graph: Graph = {}
        graph[tp] = (provider, args)

        subgraph = self.build(value_type)
        path = find_nodes_in_paths(subgraph, value_type, index_name)
        for key, value in subgraph.items():
            if key in path:
                for i in range(size):
                    provider, args = value
                    args_with_index = tuple(
                        _indexed_key(index_name, i, arg) if arg in path else arg
                        for arg in args
                    )
                    graph[_indexed_key(index_name, i, key)] = (
                        provider,
                        args_with_index,
                    )
            else:
                graph[key] = value
        for i in range(len(self._indices[index_name])):
            provider, _ = self._get_provider(Label(index_name, i))
            graph[Label(index_name, i)] = (provider, ())
        return graph

    def _make_mapping_provider(
        self, value_type: type, mapping_type: type, index_name: type
    ) -> Callable[..., Any]:
        def provider(*args):  # type: ignore[no-untyped-def]
            return mapping_type(dict(zip(self._indices[index_name], args)))

        return provider

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
