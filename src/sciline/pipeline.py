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


class ReplicatorBase(Generic[IndexType]):
    def __init__(self, index_name: type, index: Iterable[IndexType], path: List[Key]):
        self._index_name = index_name
        self.index = index
        self._path = path

    def __contains__(self, key: Key) -> bool:
        return key in self._path

    def replicate(
        self,
        key: Key,
        value: Any,
        get_provider: Callable[..., Tuple[Provider, Dict[TypeVar, Key]]],
    ) -> Graph:
        graph: Graph = {}
        provider, args = value
        for idx in self.index:
            subkey = self.key(idx, key)
            if provider == _param_sentinel:
                graph[subkey] = (get_provider(subkey)[0], ())
            else:
                graph[subkey] = self._copy_node(key, provider, args, idx)
        return graph

    def _copy_node(
        self,
        key: Key,
        provider: Union[Provider, SeriesProvider[IndexType]],
        args: Tuple[Key, ...],
        idx: IndexType,
    ) -> Tuple[Provider, Tuple[Key, ...]]:
        return (
            provider,
            tuple(self.key(idx, arg) if arg in self else arg for arg in args),
        )

    def key(self, i: IndexType, value_name: Union[Type[T], Item[T]]) -> Item[T]:
        label = Label(self._index_name, i)
        if isinstance(value_name, Item):
            return Item(value_name.label + (label,), value_name.tp)
        else:
            return Item((label,), value_name)


class Replicator(ReplicatorBase[IndexType]):
    r"""
    Helper for rewriting the graph to map over a given index.

    See Pipeline._build_series for context. Given a graph template, this makes a
    transformation as follows:

    S          P1[0]   P1[1]   P1[2]
    |           |       |       |
    A    ->    A[0]    A[1]    A[2]
    |           |       |       |
    B          B[0]    B[1]    B[2]

    Where S is a sentinel value, P1 are parameters from a parameter table, 0,1,2
    are indices of the param table rows, and A and B are arbitrary nodes in the graph.
    """

    def __init__(self, param_table: ParamTable, graph_template: Graph) -> None:
        index_name = param_table.row_dim
        super().__init__(
            index_name=index_name,
            index=param_table.index,
            path=_find_nodes_in_paths(graph_template, index_name),
        )


class GroupingReplicator(ReplicatorBase[LabelType], Generic[IndexType, LabelType]):
    r"""
    Helper for rewriting the graph to group by a given index.

    See Pipeline._build_series for context. Given a graph template, this makes a
    transformation as follows:

    P1[0]   P1[1]   P1[2]         P1[0]   P1[1]   P1[2]
     |       |       |             |       |       |
    A[0]    A[1]    A[2]          A[0]    A[1]    A[2]
     |       |       |             |       |       |
    B[0]    B[1]    B[2]    ->    B[0]    B[1]    B[2]
      \______|______/              \______/        |
             |                         |           |
            SB                       SB[x]       SB[y]
             |                         |           |
             C                        C[x]        C[y]

    Where SB is Series[Idx,B].  Here, the upper half of the graph originates from a
    prior transformation of a graph template using `Replicator`. The output of this
    combined with further nodes is the graph template passed to this class. x and y
    are the labels used in a grouping operation, based on the values of a ParamTable
    column P2.
    """

    def __init__(
        self, param_table: ParamTable, graph_template: Graph, label_name: type
    ) -> None:
        self._label_name = label_name
        self._group_node = self._find_grouping_node(param_table.row_dim, graph_template)
        self._groups: Dict[LabelType, List[IndexType]] = defaultdict(list)
        for idx, label in zip(param_table.index, param_table[label_name]):
            self._groups[label].append(idx)
        super().__init__(
            index_name=label_name,
            index=self._groups,
            path=_find_nodes_in_paths(graph_template, self._group_node),
        )

    def _copy_node(
        self,
        key: Key,
        provider: Union[Provider, SeriesProvider[IndexType]],
        args: Tuple[Key, ...],
        idx: LabelType,
    ) -> Tuple[Provider, Tuple[Key, ...]]:
        if (not isinstance(provider, SeriesProvider)) or key != self._group_node:
            return super()._copy_node(key, provider, args, idx)
        labels = self._groups[idx]
        if set(labels) - set(provider.labels):
            raise ValueError(f'{labels} is not a subset of {provider.labels}')
        selected = {
            label: arg for label, arg in zip(provider.labels, args) if label in labels
        }
        split_provider = SeriesProvider(selected, provider.row_dim)
        return (split_provider, tuple(selected.values()))

    def _find_grouping_node(self, index_name: Key, subgraph: Graph) -> type:
        ends: List[type] = []
        for key in subgraph:
            if get_origin(key) == Series and get_args(key)[0] == index_name:
                # Because of the succeeded get_origin we know it is a type
                ends.append(key)  # type: ignore[arg-type]
        if len(ends) == 1:
            return ends[0]
        raise ValueError(f"Could not find unique grouping node, found {ends}")


class SeriesProvider(Generic[KeyType]):
    """
    Internal provider for combining results obtained based on different rows in a
    param table into a single object.
    """

    def __init__(self, labels: Iterable[KeyType], row_dim: type) -> None:
        self.labels = labels
        self.row_dim = row_dim

    def __call__(self, *vals: ValueType) -> Series[KeyType, ValueType]:
        return Series(self.row_dim, dict(zip(self.labels, vals)))


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
        self._param_name_to_table_key: Dict[Key, Key] = {}
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
        if get_origin(key) == Union:
            raise ValueError('Union (or Optional) parameters are not allowed.')
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
            if param_name in self._param_name_to_table_key:
                raise ValueError(f'Parameter {param_name} already set')
        self._param_tables[params.row_dim] = params
        for param_name in params:
            self._param_name_to_table_key[param_name] = params.row_dim
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
        if get_origin(key) == Union:
            raise ValueError(
                f'Provider {provider} returning a Union (or Optional) is not allowed.'
            )
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
            if search_param_tables and tp in self._param_name_to_table_key:
                graph[tp] = (_param_sentinel, (self._param_name_to_table_key[tp],))
                continue
            if get_origin(tp) == Series:
                graph.update(self._build_series(tp))  # type: ignore[arg-type]
                continue
            if get_origin(tp) == Union:
                tps = get_args(tp)
                if len(tps) == 2 and tps[1] == type(None):  # noqa: E721
                    try:
                        optional_arg = tps[0]
                        optional_subgraph = self.build(optional_arg)
                    except UnsatisfiedRequirement:
                        graph[tp] = (lambda: None, ())
                    else:
                        optional = optional_subgraph.pop(optional_arg)
                        graph[tp] = optional
                        graph.update(optional_subgraph)
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
        """
        Build (sub)graph for a Series type implementing ParamTable-based functionality.

        We illustrate this with an example. Given a ParamTable with row_dim 'Idx':

        Idx | P1 | P2
        0   | a  | x
        1   | b  | x
        2   | c  | y

        and providers for A depending on P1 and B depending on A.  Calling
        build(Series[Idx,B]) will call _build_series(Series[Idx,B]). This results in
        the following procedure here:

        1. Call build(P1), resulting, e.g., in a graph S->A->B, where S is a sentinel.
           The sentinel is used because build() cannot find a unique P1, since it is
           not a single value but a column in a table.
        2. Instantiation of `Replicator`, which will be used to replicate the
           relevant parts of the graph (see illustration there).
        3. Insert a special `SeriesProvider` node, which will gather the duplicates of
           the 'B' node and provides the requested Series[Idx,B].
        4. Replicate the graph. Nodes that do not directly or indirectly depend on P1
           are not replicated.

        Conceptually, the final result will be {
            0: B(A(a)),
            1: B(A(b)),
            2: B(A(c))
        }.

        In more complex cases, we may be dealing with multiple levels of Series,
        which is used for grouping operations. Consider the above example, but with
        and additional provider for C depending on Series[Idx,B]. Calling
        build(Series[P2,C]) will call _build_series(Series[P2,C]). This results in
        the following procedure here:

        a. Call build(C), which results in the procedure above, i.e., a nested call
           to _build_series(Series[Idx,B]) and the resulting graph as explained above.
        b. Instantiation of `GroupingReplicator`, which will be used to replicate the
           relevant parts of the graph (see illustration there).
        c. Insert a special `SeriesProvider` node, which will gather the duplicates of
           the 'C' node and providers the requested Series[P2,C].
        c. Replicate the graph. Nodes that do not directly or indirectly depend on
           the special `SeriesProvider` node (from step 3.) are not replicated.

        Conceptually, the final result will be {
            x: C({
                0: B(A(a)),
                1: B(A(b))
            }),
            y: C({
                2: B(A(c))
            })
        }.
        """
        index_name: Type[KeyType]
        value_type: Type[ValueType]
        index_name, value_type = get_args(tp)

        subgraph = self.build(value_type, search_param_tables=True)

        replicator: ReplicatorBase[KeyType]
        if (
            index_name not in self._param_name_to_table_key
            and (params := self._param_tables.get(index_name)) is not None
        ):
            replicator = Replicator(param_table=params, graph_template=subgraph)
        elif (table_key := self._param_name_to_table_key.get(index_name)) is not None:
            replicator = GroupingReplicator(
                param_table=self._param_tables[table_key],
                graph_template=subgraph,
                label_name=index_name,
            )
        else:
            raise KeyError(f'No parameter table found for label {index_name}')

        graph: Graph = {}
        graph[tp] = (
            SeriesProvider(list(replicator.index), index_name),
            tuple(replicator.key(idx, value_type) for idx in replicator.index),
        )

        for key, value in subgraph.items():
            if key in replicator:
                graph.update(replicator.replicate(key, value, self._get_provider))
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
