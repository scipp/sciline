# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from html import escape
from typing import Any, Generator, Optional, Sequence, Tuple, TypeVar, Union

from ._utils import key_name
from .scheduler import DaskScheduler, NaiveScheduler, Scheduler
from .serialize import json_serialize_task_graph
from .typing import Graph, Item, Json, Key

T = TypeVar("T")


def _list_items(items: Sequence[str]) -> str:
    return '\n'.join(
        (
            '<ul>',
            ('\n'.join((f'<li>{escape(it)}</li>' for it in items))),
            '</ul>',
        )
    )


def _list_max_n_then_hide(items: Sequence[str], n: int = 5, header: str = '') -> str:
    def wrap(s: str) -> str:
        return '\n'.join(
            (
                '<div class="task-graph-detail-list">'
                '<style> .task-graph-detail-list ul { margin-top: 0; } </style>',
                s,
                '</div>',
            )
        )

    return wrap(
        '\n'.join(
            (
                header,
                _list_items(items),
            )
        )
        if len(items) <= n
        else '\n'.join(
            (
                '<details>',
                '<style>',
                'details[open] .task-graph-summary ul { display: none; }',
                '</style>',
                '<summary class="task-graph-summary">',
                header,
                _list_items((*items[:n], '...')),
                '</summary>',
                _list_items(items),
                '</details>',
            )
        )
    )


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
        targets: Union[type, Tuple[type, ...], Item[T], Tuple[Item[T], ...]],
        scheduler: Optional[Scheduler] = None,
    ) -> None:
        self._graph = graph
        self._keys = targets
        if scheduler is None:
            try:
                scheduler = DaskScheduler()
            except ImportError:
                scheduler = NaiveScheduler()
        self._scheduler = scheduler

    def compute(
        self,
        targets: Optional[
            Union[type, Tuple[type, ...], Item[T], Tuple[Item[T], ...]]
        ] = None,
    ) -> Any:
        """
        Compute the result of the graph.

        Parameters
        ----------
        targets:
            Optional list of keys to compute. This can be used to override the keys
            stored in the graph instance. Note that the keys must be present in the
            graph as intermediate results, otherwise KeyError is raised.

        Returns
        -------
        If ``targets`` is a single type, returns the single result that was computed.
        If ``targets`` is a tuple of types, returns a dictionary with type as keys
        and the corresponding results as values.
        """
        if targets is None:
            targets = self._keys
        if isinstance(targets, tuple):
            results = self._scheduler.get(self._graph, list(targets))
            return dict(zip(targets, results))
        else:
            return self._scheduler.get(self._graph, [targets])[0]

    def keys(self) -> Generator[Key, None, None]:
        """
        Iterate over all keys of the graph.

        Yields all keys, i.e., the types of values that can be computed or are
        provided as parameters.

        Returns
        -------
        :
            Iterable over keys.
        """
        yield from self._graph.keys()

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

    def serialize(self) -> dict[str, Json]:
        """Serialize the graph to JSON.

        See the user guide on
        `Serializing Providers <../../user-guide/serialization.rst>`_.

        Returns
        -------
        :
            A JSON object representing the graph.
        """
        return json_serialize_task_graph(self._graph)

    def _repr_html_(self) -> str:
        leafs = sorted(
            [
                escape(key_name(key))
                for key in (
                    self._keys if isinstance(self._keys, tuple) else [self._keys]
                )
            ]
        )
        roots = sorted(
            {
                escape(key_name(key))
                for key, provider in self._graph.items()
                if provider.kind != 'function'
            }
        )
        scheduler = escape(str(self._scheduler))

        def head(word: str) -> str:
            return f'<h5>{word}</h5>'

        return '\n'.join(
            (
                '<style>.task-graph-repr h5 { display: inline; }</style>',
                '<div class="task-graph-repr">',
                head('Output keys: '),
                ','.join(leafs),
                '<br>',
                head('Scheduler: '),
                scheduler,
                '<br>',
                _list_max_n_then_hide(roots, header=head('Input keys:')),
                '</div>',
            )
        )
