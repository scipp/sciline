# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping
from itertools import chain
from types import UnionType
from typing import (
    Any,
    Generic,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from sciline.task_graph import TaskGraph

from ._provider import ArgSpec, Provider, ProviderLocation, ToProvider
from ._utils import key_name
from .display import pipeline_html_repr
from .handler import (
    ErrorHandler,
    HandleAsBuildTimeException,
    HandleAsComputeTimeException,
    UnsatisfiedRequirement,
)
from .param_table import ParamTable
from .scheduler import Scheduler
from .series import Series
from .typing import Graph, Item, Key, Label

T = TypeVar('T')
KeyType = TypeVar('KeyType', bound=Key)
ValueType = TypeVar('ValueType', bound=Key)
IndexType = TypeVar('IndexType', bound=Key)
LabelType = TypeVar('LabelType', bound=Key)


class AmbiguousProvider(Exception):
    """Raised when multiple providers are found for a type."""


def _extract_typevars_from_generic_type(t: type) -> tuple[TypeVar, ...]:
    """Returns the typevars that were used in the definition of a Generic type."""
    if not hasattr(t, '__orig_bases__'):
        return ()
    return tuple(
        chain(*(get_args(b) for b in t.__orig_bases__ if get_origin(b) == Generic))
    )


def _find_all_typevars(t: type | TypeVar) -> set[TypeVar]:
    """Returns the set of all TypeVars in a type expression."""
    if isinstance(t, TypeVar):
        return {t}
    return set(chain(*map(_find_all_typevars, get_args(t))))


def _find_bounds_to_make_compatible_type(
    requested: Key,
    provided: Key | TypeVar,
) -> dict[TypeVar, Key] | None:
    """
    Check if a type is compatible to a provided type.
    If the types are compatible, return a mapping from typevars to concrete types
    that makes the provided type equal to the requested type.
    """
    if provided == requested:
        ret: dict[TypeVar, Key] = {}
        return ret
    if isinstance(provided, TypeVar):
        # If the type var has no constraints, accept anything
        if not provided.__constraints__:
            return {provided: requested}
        for c in provided.__constraints__:
            if _find_bounds_to_make_compatible_type(requested, c) is not None:
                return {provided: requested}
    if get_origin(provided) is not None:
        if get_origin(provided) == get_origin(requested):
            return _find_bounds_to_make_compatible_type_tuple(
                get_args(requested), get_args(provided)
            )
    return None


def _find_bounds_to_make_compatible_type_tuple(
    requested: tuple[Key, ...],
    provided: tuple[Key | TypeVar, ...],
) -> dict[TypeVar, Key] | None:
    """
    Check if a tuple of requested types is compatible with a tuple of provided types
    and return a mapping from type vars to concrete types that makes all provided
    types equal to their corresponding requested type.
    If any of the types is not compatible, return None.
    """
    union: dict[TypeVar, Key] = {}
    for bound in map(_find_bounds_to_make_compatible_type, requested, provided):
        # If no mapping from the type-var to a concrete type was found,
        # or if the mapping is inconsistent,
        # interrupt the search and report that no compatible types were found.
        if bound is None or any(k in union and union[k] != bound[k] for k in bound):
            return None
        union.update(bound)
    return union


def _find_all_paths(
    dependencies: Mapping[Key, Collection[Key]], start: Key, end: Key
) -> list[list[Key]]:
    """Find all paths from start to end in a DAG."""
    if start == end:
        return [[start]]
    if start not in dependencies:
        return []
    paths = []
    for node in dependencies[start]:
        if start == node:
            continue
        paths.extend(
            [start, *path] for path in _find_all_paths(dependencies, node, end)
        )
    return paths


def _find_nodes_in_paths(graph: Graph, end: Key) -> list[Key]:
    """
    Helper for Pipeline. Finds all nodes that need to be duplicated since they depend
    on a value from a param table.
    """
    start = next(iter(graph))
    dependencies = {k: tuple(p.arg_spec.keys()) for k, p in graph.items()}
    paths = _find_all_paths(dependencies, start, end)
    nodes = set()
    for path in paths:
        nodes.update(path)
    return list(nodes)


def _is_multiple_keys(
    keys: type | Iterable[type] | Item[T] | object,
) -> bool:
    # Cannot simply use isinstance(keys, Iterable) because that is True for
    # generic aliases of iterable types, e.g.,
    #
    # class Str(sl.Scope[Param, str], str): ...
    # keys = Str[int]
    #
    # And isinstance(keys, type) does not work on its own because
    # it is False for the above type.
    return (
        not isinstance(keys, type) and not get_args(keys) and isinstance(keys, Iterable)
    )


class ReplicatorBase(Generic[IndexType]):
    def __init__(self, index_name: type, index: Iterable[IndexType], path: list[Key]):
        if len(path) == 0:
            raise UnsatisfiedRequirement(
                'Could not find path to param in param table. This is likely caused '
                'by requesting a Series that does not depend directly or transitively '
                'on any param from a table.'
            )
        self._index_name = index_name
        self.index = index
        self._path = path

    def __contains__(self, key: Key) -> bool:
        return key in self._path

    def replicate(
        self,
        key: Key,
        provider: Provider,
        get_provider: Callable[..., tuple[Provider, dict[TypeVar, Key]]],
    ) -> Graph:
        graph: Graph = {}
        for idx in self.index:
            subkey = self.key(idx, key)
            if isinstance(provider, _ParamSentinel):
                graph[subkey] = get_provider(subkey)[0]
            else:
                graph[subkey] = self._copy_node(key, provider, idx)
        return graph

    def _copy_node(
        self,
        key: Key,
        provider: Provider | SeriesProvider[IndexType],
        idx: IndexType,
    ) -> Provider:
        return provider.map_arg_keys(
            lambda arg: self.key(idx, arg) if arg in self else arg
        )

    def key(self, i: IndexType, value_name: type[T] | Item[T]) -> Item[T]:
        label = Label(self._index_name, i)
        if isinstance(value_name, Item):
            return Item((*value_name.label, label), value_name.tp)
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
        self._groups: dict[LabelType, list[IndexType]] = defaultdict(list)
        for idx, label in zip(param_table.index, param_table[label_name], strict=True):
            self._groups[label].append(idx)
        super().__init__(
            index_name=label_name,
            index=self._groups,
            path=_find_nodes_in_paths(graph_template, self._group_node),
        )

    def _copy_node(
        self,
        key: Key,
        provider: Provider | SeriesProvider[IndexType],
        idx: LabelType,
    ) -> Provider:
        if (not isinstance(provider, SeriesProvider)) or key != self._group_node:
            return super()._copy_node(key, provider, idx)
        labels = self._groups[idx]
        if set(labels) - set(provider.labels):
            raise ValueError(f'{labels} is not a subset of {provider.labels}')
        if tuple(provider.arg_spec.kwargs):
            raise RuntimeError(
                'A Series was provided with keyword arguments. This should not happen '
                'and is an internal error of Sciline.'
            )
        selected = {
            label: arg
            for label, arg in zip(provider.labels, provider.arg_spec.args, strict=True)
            if label in labels
        }
        return SeriesProvider(selected.keys(), provider.row_dim, args=selected.values())

    def _find_grouping_node(self, index_name: Key, subgraph: Graph) -> type:
        ends: list[type] = [
            key
            for key in subgraph
            # Because of the succeeded get_origin we know it is a type
            if get_origin(key) == Series and get_args(key)[0] == index_name
        ]
        if len(ends) == 1:
            return ends[0]
        raise ValueError(f"Could not find unique grouping node, found {ends}")


class SeriesProvider(Generic[KeyType], Provider):
    """
    Internal provider for combining results obtained based on different rows in a
    param table into a single object.
    """

    def __init__(
        self,
        labels: Iterable[KeyType],
        row_dim: type[KeyType],
        *,
        args: Iterable[Key] | None = None,
    ) -> None:
        super().__init__(
            func=self._call,
            arg_spec=ArgSpec.from_args(*(args if args is not None else labels)),
            kind='series',
        )
        self.labels = labels
        self.row_dim = row_dim

    def _call(self, *vals: ValueType) -> Series[KeyType, ValueType]:
        return Series(self.row_dim, dict(zip(self.labels, vals, strict=True)))


class _ParamSentinel(Provider):
    def __init__(self, key: Key) -> None:
        super().__init__(
            func=lambda: None,
            arg_spec=ArgSpec.from_args(key),
            kind='sentinel',
            location=ProviderLocation(
                name=f'param_sentinel({type(key).__name__})', module='sciline'
            ),
        )


class Pipeline:
    """A container for providers that can be assembled into a task graph."""

    def __init__(
        self,
        providers: Iterable[ToProvider | Provider] | None = None,
        *,
        params: dict[type[Any], Any] | None = None,
    ):
        """
        setup a Pipeline from a list providers

        Parameters
        ----------
        providers:
            list of callable providers. Each provides its return value.
            Their arguments and return value must be annotated with type hints.
        params:
            dictionary of concrete values to provide for types.
        """
        self._providers: dict[Key, Provider] = {}
        self._subproviders: dict[type, dict[tuple[Key | TypeVar, ...], Provider]] = {}
        self._param_tables: dict[Key, ParamTable] = {}
        self._param_name_to_table_key: dict[Key, Key] = {}
        for provider in providers or []:
            self.insert(provider)
        for tp, param in (params or {}).items():
            self[tp] = param

    def insert(self, provider: ToProvider | Provider, /) -> None:
        """
        Add a callable that provides its return value to the pipeline.

        Parameters
        ----------
        provider:
            Either a callable that provides its return value. Its arguments
            and return value must be annotated with type hints.
            Or a ``Provider`` object that has been constructed from such a callable.
        """
        if not isinstance(provider, Provider):
            provider = Provider.from_function(provider)
        self._set_provider(provider.deduce_key(), provider)

    def __setitem__(self, key: type[T], param: T) -> None:
        """
        Provide a concrete value for a type.

        Parameters
        ----------
        key:
            Type to provide a value for.
        param:
            Concrete value to provide.
        """
        self._set_provider(key, Provider.parameter(param))

    def set_param_table(self, params: ParamTable) -> None:
        """
        set a parameter table for a row dimension.

        Values in the parameter table provide concrete values for a type given by the
        respective column header.

        A pipeline can have multiple parameter tables, but only one per row dimension.
        Column names must be unique across all parameter tables.

        Parameters
        ----------
        params:
            Parameter table to set.
        """
        for param_name in params:
            if (existing := self._param_name_to_table_key.get(param_name)) is not None:
                if (
                    existing == params.row_dim
                    and param_name in self._param_tables[existing]
                ):
                    # Column will be removed by del_param_table below, clash is ok
                    continue
                raise ValueError(f'Parameter {param_name} already set')
        if params.row_dim in self._param_tables:
            self.del_param_table(params.row_dim)
        self._param_tables[params.row_dim] = params
        for param_name in params:
            self._param_name_to_table_key[param_name] = params.row_dim
        for param_name, values in params.items():
            for index, label in zip(params.index, values, strict=True):
                self._set_provider(
                    Item((Label(tp=params.row_dim, index=index),), param_name),
                    Provider.table_cell(label),
                )
        for index, label in zip(params.index, params.index, strict=True):
            self._set_provider(
                Item((Label(tp=params.row_dim, index=index),), params.row_dim),
                Provider.table_cell(label),
            )

    def del_param_table(self, row_dim: type) -> None:
        """
        Remove a parameter table.

        Parameters
        ----------
        row_dim:
            Row dimension of the parameter table to remove.
        """
        # 1. Remove providers pointing to table cells
        params = self._param_tables[row_dim]
        for index in params.index:
            label = (Label(tp=row_dim, index=index),)
            for param_name in params:
                del self._providers[Item(label, param_name)]
            del self._providers[Item(label, row_dim)]
        # 2. Remove column to table mapping
        for param_name in list(self._param_name_to_table_key):
            if self._param_name_to_table_key[param_name] == row_dim:
                del self._param_name_to_table_key[param_name]
        # 3. Remove table
        del self._param_tables[row_dim]

    def set_param_series(self, row_dim: type, index: Collection[Any]) -> None:
        """
        set a series of parameters.

        This is a convenience method for creating and setting a parameter table with
        no columns and an index given by `index`.

        Parameters
        ----------
        row_dim:
            Row dimension of the parameter table to set.
        index:
            Index of the parameter table to set.
        """
        self.set_param_table(ParamTable(row_dim, columns={}, index=index))

    def _set_provider(
        self,
        key: Key,
        provider: Provider,
    ) -> None:
        # isinstance does not work here and types.NoneType available only in 3.10+
        if key == type(None):
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
            subproviders[args] = provider
        else:
            self._providers[key] = provider

    def _get_provider(
        self, tp: type[T] | Item[T], handler: ErrorHandler | None = None
    ) -> tuple[Provider, dict[TypeVar, Key]]:
        handler = handler or HandleAsBuildTimeException()
        explanation: list[str] = []
        if (provider := self._providers.get(tp)) is not None:
            return provider, {}
        elif (origin := get_origin(tp)) is not None and (
            subproviders := self._subproviders.get(origin)
        ) is not None:
            requested = get_args(tp)
            matches = [
                (subprovider, bound)
                for args, subprovider in subproviders.items()
                if (
                    bound := _find_bounds_to_make_compatible_type_tuple(requested, args)
                )
                is not None
            ]
            typevar_counts = [len(bound) for _, bound in matches]
            min_typevar_count = min(typevar_counts, default=0)
            matches = [
                m
                for count, m in zip(typevar_counts, matches, strict=True)
                if count == min_typevar_count
            ]

            if len(matches) == 1:
                provider, bound = matches[0]
                return provider, bound
            elif len(matches) > 1:
                matching_providers = [provider.location.name for provider, _ in matches]
                raise AmbiguousProvider(
                    f"Multiple providers found for type {tp}."
                    f" Matching providers are: {matching_providers}."
                )
            else:
                typevars_in_expression = _extract_typevars_from_generic_type(origin)
                if typevars_in_expression:
                    explanation = [
                        ''.join(
                            map(
                                str,
                                (
                                    'Note that ',
                                    key_name(origin[typevars_in_expression]),
                                    ' has constraints ',
                                    (
                                        {
                                            key_name(tv): tuple(
                                                map(key_name, tv.__constraints__)
                                            )
                                            for tv in typevars_in_expression
                                        }
                                    ),
                                ),
                            )
                        )
                    ]
        return handler.handle_unsatisfied_requirement(tp, *explanation), {}

    def build(
        self,
        tp: type[T] | Item[T],
        /,
        *,
        handler: ErrorHandler,
        search_param_tables: bool = False,
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
        stack: list[type[T] | Item[T]] = [tp]
        while stack:
            tp = stack.pop()
            # First look in column labels of param tables
            if search_param_tables and tp in self._param_name_to_table_key:
                graph[tp] = _ParamSentinel(self._param_name_to_table_key[tp])
                continue
            # Then also indices of param tables. This comes second because we need to
            # prefer column labels over indices for multi-level grouping.
            if search_param_tables and tp in self._param_tables:
                graph[tp] = _ParamSentinel(tp)
                continue
            if get_origin(tp) == Series:
                sub = self._build_series(tp, handler=handler)  # type: ignore[arg-type]
                graph.update(sub)
                continue
            provider, bound = self._get_provider(tp, handler=handler)
            provider = provider.bind_type_vars(bound)
            graph[tp] = provider
            stack.extend(provider.arg_spec.keys() - graph.keys())
        return graph

    def _build_series(
        self, tp: type[Series[KeyType, ValueType]], handler: ErrorHandler
    ) -> Graph:
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
        index_name: type[KeyType]
        value_type: type[ValueType]
        index_name, value_type = get_args(tp)

        subgraph = self.build(value_type, search_param_tables=True, handler=handler)

        replicator: ReplicatorBase[KeyType]
        if (
            # For multi-level grouping a type is an index as well as a column label.
            # In this case we do not want to replicate the graph, but group it (elif).
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

        graph: Graph = {
            tp: SeriesProvider(
                list(replicator.index),
                index_name,
                args=(replicator.key(idx, value_type) for idx in replicator.index),
            )
        }

        for key, provider in subgraph.items():
            if key in replicator:
                graph.update(replicator.replicate(key, provider, self._get_provider))
            else:
                graph[key] = provider
        return graph

    @overload
    def compute(self, tp: type[T], **kwargs: Any) -> T: ...

    @overload
    def compute(self, tp: Iterable[type[T]], **kwargs: Any) -> dict[type[T], T]: ...

    @overload
    def compute(self, tp: Item[T], **kwargs: Any) -> T: ...

    @overload
    def compute(self, tp: UnionType, **kwargs: Any) -> Any: ...

    def compute(
        self, tp: type | Iterable[type] | Item[T] | UnionType, **kwargs: Any
    ) -> Any:
        """
        Compute result for the given keys.

        Equivalent to ``self.get(tp).compute()``.

        Parameters
        ----------
        tp:
            Type to compute the result for.
            Can be a single type or an iterable of types.
        kwargs:
            Keyword arguments passed to the ``.get()`` method.
        """
        return self.get(tp, **kwargs).compute()

    def visualize(self, tp: type | Iterable[type], **kwargs: Any) -> graphviz.Digraph:  # type: ignore[name-defined] # noqa: F821
        """
        Return a graphviz Digraph object representing the graph for the given keys.

        Equivalent to ``self.get(tp).visualize()``.

        Parameters
        ----------
        tp:
            Type to visualize the graph for.
            Can be a single type or an iterable of types.
        kwargs:
            Keyword arguments passed to :py:class:`graphviz.Digraph`.
        """
        return self.get(tp, handler=HandleAsComputeTimeException()).visualize(**kwargs)

    def get(
        self,
        keys: type | Iterable[type] | Item[T] | object,
        *,
        scheduler: Scheduler | None = None,
        handler: ErrorHandler | None = None,
    ) -> TaskGraph:
        """
        Return a TaskGraph for the given keys.

        Parameters
        ----------
        keys:
            Type to compute the result for.
            Can be a single type or an iterable of types.
        scheduler:
            Optional scheduler to use for computing the result. If not given, a
            :py:class:`NaiveScheduler` is used if `dask` is not installed,
            otherwise dask's threaded scheduler is used.
        handler:
            Handler for unsatisfied requirements. If not provided,
            :py:class:`HandleAsBuildTimeException` is used, which raises an exception.
            During development and debugging it can be helpful to use a handler that
            raises an exception only when the graph is computed. This can be achieved
            by passing :py:class:`HandleAsComputeTimeException` as the handler.
        """
        handler = handler or HandleAsBuildTimeException()
        if _is_multiple_keys(keys):
            keys = tuple(keys)  # type: ignore[arg-type]
            graph: Graph = {}
            for t in keys:  # type: ignore[var-annotated]
                graph.update(self.build(t, handler=handler))
        else:
            graph = self.build(keys, handler=handler)  # type: ignore[arg-type]
        return TaskGraph(
            graph=graph,
            targets=keys,
            scheduler=scheduler,  # type: ignore[arg-type]
        )

    @overload
    def bind_and_call(self, fns: Callable[..., T], /) -> T: ...

    @overload
    def bind_and_call(
        self, fns: Iterable[Callable[..., Any]], /
    ) -> tuple[Any, ...]: ...

    def bind_and_call(
        self, fns: Callable[..., Any] | Iterable[Callable[..., Any]], /
    ) -> Any:
        """
        Call the given functions with arguments provided by the pipeline.

        Parameters
        ----------
        fns:
            Functions to call.
            The pipeline will provide all arguments based on the function's type hints.

            If this is a single callable, it is called directly.
            Otherwise, ``bind_and_call`` will iterate over it and call all functions.
            If will in either case call :meth:`Pipeline.compute` only once.

        Returns
        -------
        :
            The return values of the functions in the same order as the functions.
            If only one function is passed, its return value
            is *not* wrapped in a tuple.
        """
        return_tuple = True
        if callable(fns):
            fns = (fns,)
            return_tuple = False

        arg_types_per_function = {
            fn: {
                name: ty for name, ty in get_type_hints(fn).items() if name != 'return'
            }
            for fn in fns
        }
        all_arg_types = tuple(
            set(chain(*(a.values() for a in arg_types_per_function.values())))
        )
        values_per_type = self.compute(all_arg_types)
        results = tuple(
            fn(**{name: values_per_type[ty] for name, ty in arg_types.items()})
            for fn, arg_types in arg_types_per_function.items()
        )
        if not return_tuple:
            return results[0]
        return results

    def copy(self) -> Pipeline:
        """
        Make a copy of the pipeline.
        """
        out = Pipeline()
        out._providers = self._providers.copy()
        out._subproviders = {k: v.copy() for k, v in self._subproviders.items()}
        out._param_tables = self._param_tables.copy()
        out._param_name_to_table_key = self._param_name_to_table_key.copy()
        return out

    def __copy__(self) -> Pipeline:
        return self.copy()

    def _repr_html_(self) -> str:
        providers_without_parameters = (
            (origin, (), value) for origin, value in self._providers.items()
        )  # type: ignore[var-annotated]
        providers_with_parameters = (
            (origin, args, value)
            for origin in self._subproviders
            for args, value in self._subproviders[origin].items()
        )
        return pipeline_html_repr(
            chain(providers_without_parameters, providers_with_parameters)
        )
