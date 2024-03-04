# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import importlib.resources
import json
from typing import Union

from .._provider import Provider
from .._utils import key_full_qualname, key_name, provider_full_qualname, provider_name
from ..typing import Graph, Json, Key


def json_schema() -> Json:
    """Return the JSON schema for serialized task graphs.

    See, e.g., `jsonschema <https://python-jsonschema.readthedocs.io/en/stable/>`_
    for a tool that can validate the schema.

    Returns
    -------
    :
        The graph JSON schema as a dict.
    """
    with (
        importlib.resources.files('sciline.serialize')
        .joinpath('graph_json_schema.json')
        .open()
    ) as f:
        return json.load(f)  # type: ignore[no-any-return]


def json_serialize_task_graph(graph: Graph) -> dict[str, Json]:
    """Serialize a graph to JSON.

    See the user guide on
    `Serializing Providers <../../user-guide/serialization.rst>`_.

    Also available as :meth:`sciline.TaskGraph.serialize`.

    Returns
    -------
    :
        A JSON object representing the graph.
    """
    id_gen = _IdGenerator()
    nodes = []
    edges = []
    for key, provider in graph.items():
        n, e = _serialize_provider(key, provider, id_gen)
        nodes.append(n)
        edges.extend(e)
    return {
        'directed': True,
        'multigraph': False,
        'nodes': nodes,  # type: ignore[dict-item]
        'edges': edges,  # type: ignore[dict-item]
    }


def _serialize_provider(
    key: Key, provider: Provider, id_gen: _IdGenerator
) -> tuple[dict[str, Json], list[dict[str, Json]]]:
    if provider.kind == 'function':
        return _serialize_function(key, provider, id_gen)
    if provider.kind == 'parameter':
        return _serialize_param(key, id_gen)
    raise ValueError(
        f'Cannot serialize a task graph that contains {provider.kind} nodes.'
    )


def _serialize_param(
    key: Key, id_gen: _IdGenerator
) -> tuple[dict[str, Json], list[dict[str, Json]]]:
    node = {
        'id': id_gen.node_id(key),
        'kind': 'parameter',
        'label': key_name(key),
        'out': key_full_qualname(key),
    }
    return node, []  # type: ignore[return-value]


def _serialize_function(
    key: Key, provider: Provider, id_gen: _IdGenerator
) -> tuple[dict[str, Json], list[dict[str, Json]]]:
    node_id = id_gen.node_id(key)

    edges = []
    args = []
    kwargs = {}
    for i, arg in enumerate(provider.arg_spec.args):
        edge = _serialize_edge(arg, key, i, id_gen)
        edges.append(edge)
        args.append(edge['id'])
    for name, kwarg in provider.arg_spec.kwargs:
        edge = _serialize_edge(kwarg, key, name, id_gen)
        edges.append(edge)
        kwargs[name] = edge['id']

    node = {
        'id': node_id,
        'kind': 'function',
        'label': provider_name(provider),
        'function': provider_full_qualname(provider),
        'out': key_full_qualname(key),
        'args': args,
        'kwargs': kwargs,
    }

    return node, edges  # type: ignore[return-value]


def _serialize_edge(
    source: Key, target: Key, arg: Union[int, str], id_gen: _IdGenerator
) -> dict[str, str]:
    return {
        'id': id_gen.edge_id(arg, target),
        'source': id_gen.node_id(source),
        'target': id_gen.node_id(target),
    }


class _IdGenerator:
    def __init__(self) -> None:
        self._assigned: dict[int, str] = {}
        self._next = 0

    def node_id(self, key: Key) -> str:
        # Use the key instead of the provider to avoid problems with
        # unhashable providers, keys need to be hashable to construct TaskGraph.
        return self._get_or_insert(hash(key))

    def edge_id(self, arg: Union[int, str], target: Key) -> str:
        # Uses the arg number or kwarg name instead of the source key
        # to disambiguate arguments with the same type.
        return self._get_or_insert(hash((arg, target)))

    def _get_or_insert(self, hsh: int) -> str:
        try:
            return self._assigned[hsh]
        except KeyError:
            self._assigned[hsh] = str(self._next)
            self._next += 1
            return self._assigned[hsh]
