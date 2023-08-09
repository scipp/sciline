# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections import defaultdict
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
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
from .param_table import ParamTable
from .scheduler import Scheduler
from .series import Series
from .typing import Graph, Item, Key, Label, Provider

T = TypeVar('T')
KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')
IndexType = TypeVar('IndexType')
LabelType = TypeVar('LabelType')


class UnsatisfiedRequirement(Exception):
    """Raised when a type cannot be provided."""


class UnboundTypeVar(Exception):
    """
    Raised when a parameter of a generic provider is not bound to a concrete type.
    """


class AmbiguousProvider(Exception):
    """Raised when multiple providers are found for a type."""


def _indexed_key(
    index_name: Any, i: int, value_name: Union[Type[T], Item[T]]
) -> Item[T]:
    label = Label(index_name, i)
    if isinstance(value_name, Item):
        return Item(value_name.label + (label,), value_name.tp)
    else:
        return Item((label,), value_name)


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


def _find_all_paths(
    dependencies: Mapping[T, Collection[T]], start: T, end: T
) -> List[List[T]]:
    """Find all paths from start to end in a DAG."""
    if start == end:
        return [[start]]
    if start not in dependencies:
        return []
    paths = []
    for node in dependencies[start]:
        for path in _find_all_paths(dependencies, node, end):
            paths.append([start] + path)
    return paths


def _find_nodes_in_paths(
    graph: Mapping[T, Tuple[Callable[..., Any], Collection[T]]], end: T
) -> List[T]:
    """
    Helper for Pipeline. Finds all nodes that need to be duplicated since they depend
    on a value from a param table.
    """
    start = next(iter(graph))
    # 0 is the provider, 1 is the args
    dependencies = {k: v[1] for k, v in graph.items()}
    paths = _find_all_paths(dependencies, start, end)
    nodes = set()
    for path in paths:
        nodes.update(path)
    return list(nodes)


class Grouper(Protocol):
    """Helper protocol for rewriting graphs."""

    def __contains__(self, key: Any) -> bool:
        ...

    def duplicate(self, key: Any, value: Any) -> Dict[Key, Any]:
        ...


class NoGrouping(Generic[IndexType]):
    """Helper for rewriting the graph to map over a given index."""

    def __init__(self, param_table: ParamTable, graph_template: Graph) -> None:
        self.index = param_table.index
        self._index_name = param_table.row_dim
        self._path = _find_nodes_in_paths(graph_template, self._index_name)

    def __contains__(self, key: Any) -> bool:
        return key in self._path

    def duplicate(self, key: Any, value: Any, get_provider) -> Dict[Key, Any]:
        return duplicate_node(
            key, value, get_provider, self._index_name, self.index, self._path
        )


def duplicate_node(
    key: Any, value: Any, get_provider, index_name: Any, index: Any, path
) -> Dict[Key, Any]:
    graph = {}
    for idx in index:
        provider, args = value
        subkey = _indexed_key(index_name, idx, key)
        if provider == _param_sentinel:
            provider, _ = get_provider(subkey)
            args = ()
        args_with_index = tuple(
            _indexed_key(index_name, idx, arg) if arg in path else arg for arg in args
        )
        graph[subkey] = (provider, args_with_index)
    return graph


class GroupBy(Generic[IndexType, LabelType]):
    """Helper for rewriting the graph to group by a given index."""

    def __init__(
        self, param_table: ParamTable, graph_template: Graph, label_name
    ) -> None:
        self._label_name = label_name
        self._group_node = self._find_grouping_node(param_table.row_dim, graph_template)
        self._path = _find_nodes_in_paths(graph_template, self._group_node)
        self.index: Dict[LabelType, List[IndexType]] = defaultdict(list)
        for idx, label in zip(param_table.index, param_table[label_name]):
            self.index[label].append(idx)

    def __contains__(self, key: Any) -> bool:
        return key in self._path

    def duplicate(self, key: Any, value: Any, get_provider) -> Dict[Key, Any]:
        if key != self._group_node:
            return duplicate_node(
                key, value, get_provider, self._label_name, self.index, self._path
            )
        graph = {}
        provider, args = value
        for index in self.index:
            labels = self.index[index]
            if set(labels) - set(provider.labels):
                raise ValueError(f'{labels} is not a subset of {provider.labels}')
            subkey = _indexed_key(self._label_name, index, key)
            selected = {
                label: arg
                for label, arg in zip(provider.labels, args)
                if label in labels
            }
            split_provider = SeriesProvider(selected, provider._row_dim)
            graph[subkey] = (split_provider, tuple(selected.values()))
        return graph

    def _find_grouping_node(self, index_name: Key, subgraph: Graph) -> type:
        ends: List[type] = []
        for key in subgraph:
            if get_origin(key) == Series and get_args(key)[0] == index_name:
                # Because of the succeeded get_origin we know it is a type
                ends.append(key)  # type: ignore[arg-type]
        if len(ends) == 1:
            return ends[0]
        raise ValueError(f"Could not find unique grouping node, found {ends}")


class SeriesProvider(Generic[KeyType, ValueType]):
    """
    Internal provider for combining results obtained based on different rows in a
    param table into a single object.
    """

    def __init__(self, labels: Iterable[KeyType], row_dim: type) -> None:
        self.labels = labels
        self._row_dim = row_dim

    def __call__(self, *vals: ValueType) -> Series[KeyType, ValueType]:
        return Series(self._row_dim, dict(zip(self.labels, vals)))


class _param_sentinel:
    ...


class Pipeline:
    """A container for providers that can be assembled into a task graph."""

    def __init__(
        self,
        providers: Optional[List[Provider]] = None,
        *,
        params: Optional[Dict[type, Any]] = None,
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
        self._param_index_name: Dict[Key, Key] = {}
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

    def set_param_table(self, params: ParamTable) -> None:
        """
        Set a parameter table for a row dimension.

        Values in the parameter table provide concrete values for a type given by the
        respective column header.

        A pipeline can have multiple parameter tables, but only one per row dimension.
        Column names must be unique across all parameter tables.

        Parameters
        ----------
        params:
            Parameter table to set.
        """
        if params.row_dim in self._param_tables:
            raise ValueError(f'Parameter table for {params.row_dim} already set')
        for param_name in params:
            if param_name in self._param_index_name:
                raise ValueError(f'Parameter {param_name} already set')
        self._param_tables[params.row_dim] = params
        for param_name in params:
            self._param_index_name[param_name] = params.row_dim
        for param_name, values in params.items():
            for index, label in zip(params.index, values):
                self._set_provider(
                    Item((Label(tp=params.row_dim, index=index),), param_name),
                    lambda label=label: label,
                )

    def _set_provider(
        self, key: Union[Type[T], Item[T]], provider: Callable[..., T]
    ) -> None:
        # isinstance does not work here and types.NoneType available only in 3.10+
        if key == type(None):  # noqa: E721
            raise ValueError(f'Provider {provider} returning `None` is not allowed')
        if get_origin(key) == Series:
            raise ValueError(
                f'Provider {provider} returning a sciline.Series is not allowed. '
                'Series is a special container reserved for use in conjunction with '
                'sciline.ParamTable and must not be provided directly.'
            )
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
        self, tp: Union[Type[T], Item[T]]
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

    def build(
        self, tp: Union[Type[T], Item[T]], /, search_param_tables: bool = False
    ) -> Graph:
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
        search_param_tables:
            Whether to search parameter tables for concrete keys.
        """
        graph: Graph = {}
        stack: List[Union[Type[T], Item[T]]] = [tp]
        while stack:
            tp = stack.pop()
            if search_param_tables and tp in self._param_index_name:
                graph[tp] = (_param_sentinel, (self._param_index_name[tp],))
                continue
            if get_origin(tp) == Series:
                graph.update(self._build_series(tp))  # type: ignore[arg-type]
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

    def _build_series(self, tp: Type[Series[KeyType, ValueType]]) -> Graph:
        label_name: Type[KeyType]
        value_type: Type[ValueType]
        label_name, value_type = get_args(tp)
        # Step 1:
        # Build a graph that can compute the value type. As we are building
        # a Series, this will terminate when it reaches a parameter that is not a
        # single provided value but a collection of values from a parameter table
        # column. Instead of single value (which does not exist), a sentinel is
        # used to mark this, for processing below.
        subgraph = self.build(value_type, search_param_tables=True)
        # Step 2:
        # Identify nodes in the graph that need to be duplicated as they lie in the
        # path to a parameter from a table. In the case of grouping, note that the
        # ungrouped graph (including duplication of nodes) will have been built by a
        # prior call to _build_series, so instead of duplicating everything until the
        # param table is reached, we only duplicate until the node that is performing
        # the grouping.
        grouper: Grouper
        if (
            label_name not in self._param_index_name
            and (params := self._param_tables.get(label_name)) is not None
        ):
            grouper = NoGrouping(param_table=params, graph_template=subgraph)
        elif (index_name := self._param_index_name.get(label_name)) is not None:
            grouper = GroupBy(
                param_table=self._param_tables[index_name],
                graph_template=subgraph,
                label_name=label_name,
            )
        else:
            raise KeyError(f'No parameter table found for label {label_name}')

        graph: Graph = {}
        graph[tp] = (
            SeriesProvider(list(grouper.index), label_name),
            tuple(_indexed_key(label_name, idx, value_type) for idx in grouper.index),
        )

        # Step 3:
        # Duplicate nodes, replacing keys with indexed keys.
        for key, value in subgraph.items():
            if key in grouper:
                graph.update(grouper.duplicate(key, value, self._get_provider))
            else:
                graph[key] = value
        return graph

    @overload
    def compute(self, tp: Type[T]) -> T:
        ...

    @overload
    def compute(self, tp: Tuple[Type[T], ...]) -> Tuple[T, ...]:
        ...

    @overload
    def compute(self, tp: Item[T]) -> T:
        ...

    def compute(self, tp: type | Tuple[type, ...] | Item[T]) -> Any:
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
        self,
        keys: type | Tuple[type, ...] | Item[T],
        *,
        scheduler: Optional[Scheduler] = None,
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
