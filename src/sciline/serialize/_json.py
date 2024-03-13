# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import importlib.resources
import json
from typing import Any, Union

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
        nodes.extend(n)
        edges.extend(e)
    return {
        'directed': True,
        'multigraph': False,
        'nodes': nodes,  # type: ignore[dict-item]
        'edges': edges,  # type: ignore[dict-item]
    }


# returns tuple[list[node], list[edge]]
# where the nodes are either a single data node or a data node and a function node
def _serialize_provider(
    key: Key, provider: Provider, id_gen: _IdGenerator
) -> tuple[list[dict[str, Json]], list[dict[str, Json]]]:
    if provider.kind == 'function':
        return _serialize_function(key, provider, id_gen)
    if provider.kind == 'parameter':
        return _serialize_param(key, id_gen)
    raise ValueError(
        f'Cannot serialize a task graph that contains {provider.kind} nodes.'
    )


def _serialize_param(
    key: Key, id_gen: _IdGenerator
) -> tuple[list[dict[str, Json]], list[dict[str, Json]]]:
    node = {
        'id': id_gen.data_node_id(key),
        'kind': 'data',
        'label': key_name(key),
        'type': key_full_qualname(key),
    }
    return [node], []  # type: ignore[list-item]


def _serialize_function(
    key: Key, provider: Provider, id_gen: _IdGenerator
) -> tuple[list[dict[str, Json]], list[dict[str, Json]]]:
    edges = []
    args = []
    kwargs = {}
    for i, arg in enumerate(provider.arg_spec.args):
        edge = _serialize_edge_data_to_fn(arg, key, i, id_gen)
        edges.append(edge)
        args.append(edge['id'])
    for name, kwarg in provider.arg_spec.kwargs:
        edge = _serialize_edge_data_to_fn(kwarg, key, name, id_gen)
        edges.append(edge)
        kwargs[name] = edge['id']
    edges.append(_serialize_edge_fn_to_data(key, key, id_gen))

    fn_node = {
        'id': id_gen.function_node_id(key),
        'kind': 'function',
        'label': _provider_name(provider),
        'function': provider_full_qualname(provider),
        'args': args,
        'kwargs': kwargs,
    }
    [data_node], _ = _serialize_param(key, id_gen)

    return [fn_node, data_node], edges  # type: ignore[list-item, return-value]


def _serialize_edge_data_to_fn(
    source: Key, target: Key, arg: Union[int, str], id_gen: _IdGenerator
) -> dict[str, str]:
    return {
        'id': id_gen.edge_id(source, target, arg),
        'source': id_gen.data_node_id(source),
        'target': id_gen.function_node_id(target),
    }


def _serialize_edge_fn_to_data(
    source: Key, target: Key, id_gen: _IdGenerator
) -> dict[str, str]:
    return {
        'id': id_gen.edge_id(source, target, 0),
        'source': id_gen.function_node_id(source),
        'target': id_gen.data_node_id(target),
    }


def _provider_name(provider: Provider) -> str:
    try:
        return provider_name(provider)
    except AttributeError:
        # E.g. with callable objects:
        # AttributeError: '<class>' object has no attribute '__name__'
        raise ValueError(
            f"Unsupported provider for serializing graph: '{provider}' "
            'Callable objects cannot be serialized.'
        )


class _IdGenerator:
    def __init__(self) -> None:
        self._assigned: dict[Any, str] = {}
        self._next = 0

    def data_node_id(self, key: Key) -> str:
        # Keys must be unique and are required to be hashable to construct TaskGraph.
        return self._get_or_insert(key)

    def function_node_id(self, key: Key) -> str:
        # Use the key instead of the provider to avoid problems with
        # unhashable providers.
        # Use tuple with 'fn' to disambiguate from data nodes.
        return self._get_or_insert(('fn', key))

    def edge_id(self, source: Key, target: Key, arg: Union[int, str]) -> str:
        # Uses the arg number or kwarg name
        # to disambiguate arguments with the same type.
        return self._get_or_insert((source, target, arg))

    def _get_or_insert(self, hashable: Any) -> str:
        try:
            return self._assigned[hashable]
        except KeyError:
            id_ = str(self._next)
            self._assigned[hashable] = id_
            self._next += 1
            return id_
