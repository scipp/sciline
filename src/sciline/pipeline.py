# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
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
from .param_table import ParamTable
from .scheduler import Graph, Scheduler
from .series import Series

T = TypeVar('T')
KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')


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

    _param_sentinel = object()

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
        self._param_tables: Dict[Key, ParamTable] = {}
        self._param_series: Dict[Key, Key] = {}
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

    @property
    def param_tables(self) -> Dict[Key, ParamTable]:
        return dict(self._param_tables)

    def set_param_table(self, params: ParamTable) -> None:
        if params.row_dim in self._param_tables:
            raise ValueError(f'Parameter table for {params.row_dim} already set')
        for param_name in params:
            if param_name in self._param_series:
                raise ValueError(f'Parameter {param_name} already set')
        self._param_tables[params.row_dim] = params
        for param_name in params:
            self._param_series[param_name] = params.row_dim
        for param_name, values in params.items():
            for i, label in enumerate(values):
                self._set_provider(
                    Item((Label(tp=params.row_dim, index=i),), param_name),
                    lambda label=label: label,
                )

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

    def build(self, tp: Type[T], /, search_param_tables: bool = False) -> Graph:
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
            if search_param_tables and tp in self._param_series:
                graph[tp] = (self._param_sentinel, (self._param_series[tp],))
                continue
            if get_origin(tp) == Series:
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

    def _build_indexed_subgraph(self, tp: Type[Series[KeyType, ValueType]]) -> Graph:
        index_name: Type[KeyType]
        value_type: Type[ValueType]
        index_name, value_type = get_args(tp)
        index = self._param_tables[index_name].index
        size = len(index)
        args = [_indexed_key(index_name, i, value_type) for i in range(size)]
        graph: Graph = {}
        graph[tp] = (lambda *values: Series(dict(zip(index, values))), args)

        subgraph = self.build(value_type, search_param_tables=True)
        path = find_nodes_in_paths(subgraph, value_type, index_name)
        for key, value in subgraph.items():
            if key in path:
                for i in range(size):
                    provider, args = value
                    subkey = _indexed_key(index_name, i, key)
                    if provider == self._param_sentinel:
                        provider, _ = self._get_provider(subkey)
                        args = ()
                    args_with_index = tuple(
                        _indexed_key(index_name, i, arg) if arg in path else arg
                        for arg in args
                    )
                    graph[subkey] = (provider, args_with_index)
            else:
                graph[key] = value
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
