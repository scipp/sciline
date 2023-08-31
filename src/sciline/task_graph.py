# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any, Optional, Tuple, TypeVar, Union

from .scheduler import DaskScheduler, NaiveScheduler, Scheduler
from .typing import Graph, Item

T = TypeVar("T")


class TaskGraph:
    """
    Holds a concrete task graph and keys to compute.

    Task graphs are typically created by :py:class:`sciline.Pipeline.build`. They allow
    for computing all or a subset of the results in the graph.
    """

    def __init__(
        self,
        *,
        graph: Graph,
        keys: Union[type, Tuple[type, ...], Item[T], Tuple[Item[T], ...]],
        scheduler: Optional[Scheduler] = None,
    ) -> None:
        self._graph = graph
        self._keys = keys
        if scheduler is None:
            try:
                scheduler = DaskScheduler()
            except ImportError:
                scheduler = NaiveScheduler()
        self._scheduler = scheduler

    def compute(
        self,
        keys: Optional[
            Union[type, Tuple[type, ...], Item[T], Tuple[Item[T], ...], None]
        ] = None,
    ) -> Any:
        """
        Compute the result of the graph.

        Parameters
        ----------
        keys:
            Optional list of keys to compute. This can be used to override the keys
            stored in the graph instance. Note that the keys must be present in the
            graph as intermediate results, otherwise KeyError is raised.

        Returns
        -------
        If ``keys`` is a single type, returns the single result that was computed.
        If ``keys`` is a tuple of types, returns a dictionary with type as keys
        and the corresponding results as values.

        """
        if keys is None:
            keys = self._keys
        if isinstance(keys, tuple):
            results = self._scheduler.get(self._graph, list(keys))
            return {tp: result for tp, result in zip(keys, results)}
        else:
            return self._scheduler.get(self._graph, [keys])[0]

    def visualize(
        self, **kwargs: Any
    ) -> graphviz.Digraph:  # type: ignore[name-defined] # noqa: F821
        """
        Return a graphviz Digraph object representing the graph.

        Parameters
        ----------
        kwargs:
            Keyword arguments passed to :py:class:`graphviz.Digraph`.
        """
        from .visualize import to_graphviz

        return to_graphviz(self._graph, **kwargs)
