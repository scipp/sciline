# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import chain
from types import UnionType
from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_type_hints, overload

from ._provider import Provider, ToProvider
from .data_graph import DataGraph, to_task_graph
from .display import pipeline_html_repr
from .handler import ErrorHandler, HandleAsComputeTimeException
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

    def visualize(self, tp: type | Iterable[type], **kwargs: Any) -> graphviz.Digraph:
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
        graph = to_task_graph(self, targets=targets, handler=handler)
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
        nodes = ((key, data) for key, data in self._graph.nodes.items())
        return pipeline_html_repr(nodes)


def get_mapped_node_names(graph: Pipeline, key: type) -> pandas.Series:
    """
    Given a graph with key depending on mapped nodes, return a series corresponding
    mapped keys.

    This is meant to be used in combination with :py:func:`Pipeline.map`.
    If the key depends on multiple indices, the series will be a multi-index series.

    Note that Pandas is not a dependency of Sciline and must be installed separately.

    Parameters
    ----------
    graph:
        The pipeline to get the mapped key names from.
    key:
        The key to get the mapped key names for. This key must depend on mapped nodes.

    Returns
    -------
    :
        The series of mapped key names.
    """
    import pandas as pd
    from cyclebane.graph import IndexValues, NodeName

    graph = graph[key]  # Drops unrelated indices
    indices = graph._cbgraph.indices
    if len(indices) == 0:
        raise ValueError(f"'{key}' does not depend on any mapped nodes.")
    index_names = tuple(indices)
    index = pd.MultiIndex.from_product(indices.values(), names=index_names)
    keys = tuple(NodeName(key, IndexValues(index_names, idx)) for idx in index)
    if index.nlevels == 1:  # Avoid more complicate MultiIndex if unnecessary
        index = index.get_level_values(0)
    return pd.Series(keys, index=index, name=key)


def compute_series(graph: Pipeline, key: type) -> pandas.Series:
    """
    Given a graph with key depending on mapped nodes, compute a series for the key.

    This is meant to be used in combination with :py:func:`Pipeline.map`.
    If the key depends on multiple indices, the series will be a multi-index series.

    Note that Pandas is not a dependency of Sciline and must be installed separately.

    Parameters
    ----------
    graph:
        The pipeline to compute the series from.
    key:
        The key to compute the series for. This key must depend on mapped nodes.

    Returns
    -------
    :
        The computed series.
    """
    key_series = get_mapped_node_names(graph, key)
    results = graph.compute(key_series)
    return key_series.apply(lambda x: results[x])
