# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from itertools import chain
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    get_args,
    get_type_hints,
    overload,
)

from ._provider import Provider, ToProvider
from ._utils import key_name
from .data_graph import DataGraph, to_task_graph
from .display import pipeline_html_repr
from .handler import ErrorHandler, HandleAsComputeTimeException, UnsatisfiedRequirement
from .scheduler import Scheduler
from .task_graph import TaskGraph
from .typing import Key

if TYPE_CHECKING:
    import graphviz
    import pandas


T = TypeVar('T')
KeyType = TypeVar('KeyType', bound=Key)


def _is_multiple_keys(keys: type | Iterable[type] | UnionType) -> bool:
    # Cannot simply use isinstance(keys, Iterable) because that is True for
    # generic aliases of iterable types, e.g.,
    #
    # class Str(sl.Scope[Param, str], str): ...
    # keys = Str[int]
    #
    # And isinstance(keys, type) does not work on its own because
    # it is False for the above type.
    if isinstance(keys, str):
        return False
    return (
        not isinstance(keys, type) and not get_args(keys) and isinstance(keys, Iterable)
    )


class Pipeline(DataGraph):
    """A container for providers that can be assembled into a task graph."""

    def __init__(
        self,
        providers: Iterable[ToProvider | Provider] | None = None,
        *,
        params: dict[type[Any], Any] | None = None,
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
        super().__init__(providers)
        for tp, param in (params or {}).items():
            self[tp] = param

    @overload
    def compute(self, tp: type[T], **kwargs: Any) -> T: ...

    @overload
    def compute(self, tp: Iterable[type[T]], **kwargs: Any) -> dict[type[T], T]: ...

    @overload
    def compute(self, tp: UnionType, **kwargs: Any) -> Any: ...

    def compute(self, tp: type | Iterable[type] | UnionType, **kwargs: Any) -> Any:
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

    def visualize(
        self,
        tp: type | Iterable[type] | None = None,
        compact: bool = False,
        mode: Literal['data', 'task', 'both'] = 'data',
        cluster_generics: bool = True,
        cluster_color: str | None = '#f0f0ff',
        **kwargs: Any,
    ) -> graphviz.Digraph:
        """
        Return a graphviz Digraph object representing the graph for the given keys.

        Equivalent to ``self.get(tp).visualize()``.

        Parameters
        ----------
        tp:
            Type to visualize the graph for.
            Can be a single type or an iterable of types.
        compact:
            If True, parameter-table-dependent branches are collapsed into a single copy
            of the branch. Recommended for large graphs with long parameter tables.
        mode:
            If 'data', only data nodes are shown. If 'task', only task nodes and input
            data nodes are shown. If 'both', all nodes are shown.
        cluster_generics:
            If True, generic products are grouped into clusters.
        cluster_color:
            Background color of clusters. If None, clusters are dotted.
        kwargs:
            Keyword arguments passed to :py:class:`graphviz.Digraph`.
        """
        if tp is None:
            tp = self.output_keys()
        return self.get(tp, handler=HandleAsComputeTimeException()).visualize(
            compact=compact,
            mode=mode,
            cluster_generics=cluster_generics,
            cluster_color=cluster_color,
            **kwargs,
        )
        return self.get(tp, handler=HandleAsComputeTimeException()).visualize(**kwargs)

    def get(
        self,
        keys: type | Iterable[type] | UnionType,
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
        if multi := _is_multiple_keys(keys):
            targets = tuple(keys)  # type: ignore[arg-type]
        else:
            targets = (keys,)  # type: ignore[assignment]
        try:
            graph = to_task_graph(self, targets=targets, handler=handler)
        except UnsatisfiedRequirement as e:
            output_keys = ", ".join(map(repr, self.output_keys()))
            raise type(e)(f'{e} Did you mean one of: {output_keys}?') from e
        return TaskGraph(
            graph=graph,
            targets=targets if multi else keys,  # type: ignore[arg-type]
            scheduler=scheduler,
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

    def _repr_html_(self) -> str:
        nodes = ((key, data) for key, data in self.underlying_graph.nodes.items())
        return pipeline_html_repr(nodes)

    def output_keys(self) -> tuple[Key, ...]:
        """Returns the keys that are not inputs to any other providers."""
        sink_nodes = [
            node for node, degree in self.underlying_graph.out_degree if degree == 0
        ]
        return tuple(sorted(sink_nodes, key=key_name))


def get_mapped_node_names(
    graph: DataGraph, base_name: type, *, index_names: Sequence[Hashable] | None = None
) -> pandas.Series:
    """
    Given a graph with a mapped node with given base_name, return a series of
    corresponding mapped names.

    This is meant to be used in combination with :py:func:`DataGraph.map`.
    If the mapped node depends on multiple indices, the index of the returned series
    will have a multi-index.

    Note that Pandas is not a dependency of Sciline and must be installed separately.

    Parameters
    ----------
    graph:
        The data graph to get the mapped node names from.
    base_name:
        The base name of the mapped node to get the names for.
    index_names:
        Specifies the names of the indices of the mapped node. If not given this is
        inferred from the graph, but the argument may be required to disambiguate
        multiple mapped nodes with the same name.

    Returns
    -------
    :
        The series of node names corresponding to the mapped node.
    """
    import pandas as pd
    from cyclebane.graph import IndexValues, MappedNode, NodeName

    candidates = [
        node
        for node in graph.underlying_graph.nodes
        if isinstance(node, MappedNode) and node.name == base_name
    ]
    if len(candidates) == 0:
        raise ValueError(f"'{base_name}' is not a mapped node.")
    if index_names is not None:
        candidates = [
            node for node in candidates if set(node.indices) == set(index_names)
        ]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple mapped nodes with name '{base_name}' found: {candidates}"
        )

    index_names = tuple(reversed(candidates[0].indices))
    indices = {name: idx for name, idx in graph.indices.items() if name in index_names}

    index = pd.MultiIndex.from_product(indices.values(), names=index_names)
    keys = tuple(NodeName(base_name, IndexValues(index_names, idx)) for idx in index)
    if index.nlevels == 1:  # Avoid more complicated MultiIndex if unnecessary
        index = index.get_level_values(0)
    return pd.Series(keys, index=index, name=base_name)


def compute_mapped(
    pipeline: Pipeline,
    base_name: type,
    *,
    index_names: Sequence[Hashable] | None = None,
) -> pandas.Series:
    """
    Given a graph with a mapped node with given base_name, return a series of computed
    results.

    This is meant to be used in combination with :py:func:`Pipeline.map`.
    If the mapped node depends on multiple indices, the index of the returned series
    will have a multi-index.

    Note that Pandas is not a dependency of Sciline and must be installed separately.

    Parameters
    ----------
    graph:
        The data graph to get the mapped node names from.
    base_name:
        The base name of the mapped node to get the names for.
    index_names:
        Specifies the names of the indices of the mapped node. If not given this is
        inferred from the graph, but the argument may be required to disambiguate
        multiple mapped nodes with the same name.

    Returns
    -------
    :
        The series of computed results corresponding to the mapped node.
    """
    key_series = get_mapped_node_names(
        graph=pipeline, base_name=base_name, index_names=index_names
    )
    results = pipeline.compute(key_series)
    return key_series.apply(lambda x: results[x])
