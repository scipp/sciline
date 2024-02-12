# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Union, get_args

from .._provider import Provider
from .._utils import key_full_qualname, key_name, provider_full_qualname, provider_name
from ..typing import Graph, Item, Json, Key


def json_serialize_task_graph(graph: Graph) -> dict[str, Json]:
    node_ids = _UniqueNodeId()
    nodes = []
    edges = []
    for key, provider in graph.items():
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
    from ..pipeline import SeriesProvider

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
