# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from html import escape
from typing import Any, Generator, Optional, Sequence, Tuple, TypeVar, Union, get_args

from ._provider import Provider
from ._utils import key_full_qualname, key_name, provider_full_qualname, provider_name
from .scheduler import DaskScheduler, NaiveScheduler, Scheduler
from .typing import Graph, Item, Key

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
            Union[type, Tuple[type, ...], Item[T], Tuple[Item[T], ...]]
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
            return dict(zip(keys, results))
        else:
            return self._scheduler.get(self._graph, [keys])[0]

    # TODO remove again?
    def nodes(self) -> Generator[Union[Key, Provider], None, None]:
        """Iterate over all nodes of the graph.

        Nodes are both keys, i.e., the types of values that can be computed
        and providers.

        Returns
        -------
        :
            Iterable over keys and providers.
        """
        for key, provider in self._graph.items():
            yield key
            yield provider

    def edges(
        self,
    ) -> Generator[Union[tuple[Key, Provider], tuple[Provider, Key]], None, None]:
        """Iterate over all edges of the graph.

        Returns
        -------
        :
            Iterable over pairs ``(source, target)`` which indicate a directed edge
            from ``source`` to ``target``.
            There are two cases:

            - ``source`` is a key, ``target`` is a provider.
            - ``source`` is a provider, ``target`` is a key.
        """
        for key, provider in self._graph.items():
            yield provider, key
            for arg in provider.arg_spec.keys():
                yield arg, provider

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

    def serialize(self) -> dict[str, Any]:
        node_ids = _UniqueNodeId()
        nodes = []
        edges = []
        for key, provider in self._graph.items():
            key_id = node_ids.get(key)
            provider_id = node_ids.get(provider)
            nodes.append(_serialize_data_node(key, key_id))
            nodes.append(_serialize_provider_node(provider, key, provider_id))

            edges.append(_serialize_edge(provider_id, key_id))
            if provider.kind in ('function', 'series'):
                for arg in provider.arg_spec.keys():
                    edges.append(_serialize_edge(node_ids.get(arg), provider_id))

        return {
            'directed': True,
            'multigraph': False,
            'nodes': nodes,
            'edges': edges,
        }

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


def _serialize_data_node(key: Key, key_id: str) -> dict[str, str]:
    if isinstance(key, Item):
        return {
            'id': key_id,
            'kind': 'data_table_cell',
            'label': key_name(key),
            'value_type': key_full_qualname(key.tp),
            'row_types': [key_full_qualname(label.tp) for label in key.label],
            'row_indices': [key_full_qualname(label.index) for label in key.label],
        }
    return {
        'id': key_id,
        'kind': 'data',
        'label': key_name(key),
        'type': key_full_qualname(key),
    }


def _serialize_provider_node(
    provider: Provider, key: Key, provider_id: str
) -> dict[str, str]:
    from .pipeline import SeriesProvider

    if isinstance(provider, SeriesProvider):
        row_dim = provider.row_dim
        value_type = get_args(key)[1]
        return {
            'id': provider_id,
            'kind': 'p_series',
            'label': f'provide_series[{key_name(row_dim)}, {key_name(value_type)}]',
            'value_type': key_full_qualname(value_type),
            'row_dim': key_full_qualname(row_dim),
            'labels': list(map(key_full_qualname, provider.labels)),
        }
    if provider.kind == 'function':
        return {
            'id': provider_id,
            'kind': 'p_function',
            'label': provider_name(provider),
            'function': provider_full_qualname(provider),
        }
    if provider.kind == 'parameter':
        return {
            'id': provider_id,
            'kind': 'p_parameter',
            'label': key_name(key),
            'type': key_full_qualname(key),
        }
    if provider.kind == 'table_cell':
        return {
            'id': provider_id,
            'kind': 'p_table_cell',
            'label': f'table_cell({key_name(key)})',
        }
    raise ValueError(
        f'Cannot serialize graph containing providers of kind {provider.kind}'
    )


def _serialize_edge(source_id: str, target_id: str) -> dict[str, str]:
    return {'source': source_id, 'target': target_id}


class _UniqueNodeId:
    def __init__(self) -> None:
        self._assigned: dict[int, str] = {}
        self._next = 0

    def get(self, obj: Union[Key, Provider]) -> str:
        hsh = hash(obj)
        try:
            return self._assigned[hsh]
        except KeyError:
            self._assigned[hsh] = str(self._next)
            self._next += 1
            return self._assigned[hsh]
