# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# type: ignore #noqa: PGH003

from copy import deepcopy
from typing import Any, NewType, TypeVar

import jsonschema
import pytest

import sciline as sl
from sciline.serialize import json_schema
from sciline.typing import Json

A = NewType('A', int)
B = NewType('B', int)
T = TypeVar('T', A, B)


class Int(sl.Scope[T, int], int): ...


class List(sl.Scope[T, list[int]], list[int]): ...


def make_int_b() -> Int[B]:
    return Int[B](2)


def zeros(n: Int[T]) -> List[T]:
    return List[T]([0] * n)


def to_string(b: List[B], a: List[A]) -> str:
    return f'a: {a}, b: {b}'


def as_float(x: int) -> float:
    return 0.5 * x


def make_graph_predictable(graph: dict[str, Json]) -> dict[str, Json]:
    """Sort nodes and edges and assign ids according to a fixed scheme.

    Nodes:
    - Sorted by ``out``.
    - Ids are counted up from 0 for the sorted nodes.

    Edges:
    - Sorted by ``source + target`` *after* remapping the node ids.
    - Ids are counted up from 100 for the sorted edges.
    """

    def node_sort_key(node: dict[str, Any]) -> str:
        if node['kind'] == 'data':
            return node['type'] + 'data'
        out_edge = next(e for e in graph['edges'] if e['source'] == node['id'])
        data_node = next(n for n in graph['nodes'] if n['id'] == out_edge['target'])
        return data_node['type'] + 'function'

    id_mapping = {}
    nodes: Any = sorted(deepcopy(graph['nodes']), key=node_sort_key)
    edges: Any = deepcopy(graph['edges'])
    for i, node in enumerate(nodes):
        new_id = str(i)
        id_mapping[node['id']] = new_id
        node['id'] = new_id

    for edge in edges:
        edge['source'] = id_mapping[edge['source']]
        edge['target'] = id_mapping[edge['target']]
    edges = sorted(edges, key=lambda e: e['source'] + e['target'])
    for i, edge in enumerate(edges, 100):
        new_id = str(i)
        id_mapping[edge['id']] = new_id
        edge['id'] = new_id

    for node in nodes:
        if node['kind'] == 'function':
            node['args'] = [id_mapping[arg] for arg in node['args']]
            node['kwargs'] = {k: id_mapping[v] for k, v in node['kwargs'].items()}

    return {**graph, 'nodes': nodes, 'edges': edges}


# Ids correspond to the result of assign_predictable_ids
expected_serialized_nodes = [
    {
        'id': '0',
        'label': 'str',
        'kind': 'data',
        'type': 'builtins.str',
    },
    {
        'id': '1',
        'label': 'to_string',
        'kind': 'function',
        'function': 'tests.serialize.json_test.to_string',
        'args': ['106', '104'],
        'kwargs': {},
    },
    {
        'id': '2',
        'label': 'Int[A]',
        'kind': 'data',
        'type': 'tests.serialize.json_test.Int[tests.serialize.json_test.A]',
    },
    {
        'id': '3',
        'label': 'Int[B]',
        'kind': 'data',
        'type': 'tests.serialize.json_test.Int[tests.serialize.json_test.B]',
    },
    {
        'id': '4',
        'label': 'make_int_b',
        'kind': 'function',
        'function': 'tests.serialize.json_test.make_int_b',
        'args': [],
        'kwargs': {},
    },
    {
        'id': '5',
        'label': 'List[A]',
        'kind': 'data',
        'type': 'tests.serialize.json_test.List[tests.serialize.json_test.A]',
    },
    {
        'id': '6',
        'label': 'zeros',
        'kind': 'function',
        'function': 'tests.serialize.json_test.zeros',
        'args': ['101'],
        'kwargs': {},
    },
    {
        'id': '7',
        'label': 'List[B]',
        'kind': 'data',
        'type': 'tests.serialize.json_test.List[tests.serialize.json_test.B]',
    },
    {
        'id': '8',
        'label': 'zeros',
        'kind': 'function',
        'function': 'tests.serialize.json_test.zeros',
        'args': ['102'],
        'kwargs': {},
    },
]
expected_serialized_edges = [
    {'id': '100', 'source': '1', 'target': '0'},
    {'id': '101', 'source': '2', 'target': '6'},
    {'id': '102', 'source': '3', 'target': '8'},
    {'id': '103', 'source': '4', 'target': '3'},
    {'id': '104', 'source': '5', 'target': '1'},
    {'id': '105', 'source': '6', 'target': '5'},
    {'id': '106', 'source': '7', 'target': '1'},
    {'id': '107', 'source': '8', 'target': '7'},
]
expected_serialized_graph = {
    'directed': True,
    'multigraph': False,
    'nodes': expected_serialized_nodes,
    'edges': expected_serialized_edges,
}


def test_serialize() -> None:
    pl = sl.Pipeline([make_int_b, zeros, to_string], params={Int[A]: 3})
    graph = pl.get(str)
    res = graph.serialize()
    res = make_graph_predictable(res)
    assert res == expected_serialized_graph


def fn_w_kwonlyargs(*, x: int) -> float:
    return 0.5 * x


# Ids correspond to the result of assign_predictable_ids
expected_serialized_kwonlyargs_nodes = [
    {
        'id': '0',
        'label': 'float',
        'kind': 'data',
        'type': 'builtins.float',
    },
    {
        'id': '1',
        'label': 'fn_w_kwonlyargs',
        'kind': 'function',
        'function': 'tests.serialize.json_test.fn_w_kwonlyargs',
        'args': [],
        'kwargs': {'x': '101'},
    },
    {
        'id': '2',
        'label': 'int',
        'kind': 'data',
        'type': 'builtins.int',
    },
]
expected_serialized_kwonlyargs_edges = [
    {'id': '100', 'source': '1', 'target': '0'},
    {'id': '101', 'source': '2', 'target': '1'},
]
expected_serialized_kwonlyargs_graph = {
    'directed': True,
    'multigraph': False,
    'nodes': expected_serialized_kwonlyargs_nodes,
    'edges': expected_serialized_kwonlyargs_edges,
}


def test_serialize_kwonlyargs() -> None:
    pl = sl.Pipeline([fn_w_kwonlyargs], params={int: 3})
    graph = pl.get(float)
    res = graph.serialize()
    res = make_graph_predictable(res)
    assert res == expected_serialized_kwonlyargs_graph


# Ids correspond to the result of assign_predictable_ids
expected_serialized_lambda_nodes = [
    {
        'id': '0',
        'label': 'float',
        'kind': 'data',
        'type': 'builtins.float',
    },
    {
        'id': '1',
        'label': '<lambda>',
        'kind': 'function',
        'function': 'tests.serialize.json_test.test_serialize_lambda.<locals>.<lambda>',
        'args': ['101'],
        'kwargs': {},
    },
    {
        'id': '2',
        'label': 'int',
        'kind': 'data',
        'type': 'builtins.int',
    },
]
expected_serialized_lambda_edges = [
    {'id': '100', 'source': '1', 'target': '0'},
    {'id': '101', 'source': '2', 'target': '1'},
]
expected_serialized_lambda_graph = {
    'directed': True,
    'multigraph': False,
    'nodes': expected_serialized_lambda_nodes,
    'edges': expected_serialized_lambda_edges,
}


def test_serialize_lambda() -> None:
    lam = lambda x: float(x)  # noqa: E731
    lam.__annotations__['x'] = int
    lam.__annotations__['return'] = float
    pl = sl.Pipeline((lam,), params={int: 4})
    graph = pl.get(float)
    res = graph.serialize()
    res = make_graph_predictable(res)
    assert res == expected_serialized_lambda_graph


def test_serialize_does_not_support_callable_objects() -> None:
    class C:
        def __call__(self, x: int) -> float:
            return float(x)

    pl = sl.Pipeline((C(),), params={int: 4})
    graph = pl.get(float)
    with pytest.raises(ValueError, match='Callable objects cannot be serialized'):
        graph.serialize()


def test_serialize_validate_schema() -> None:
    pl = sl.Pipeline([make_int_b, zeros, to_string], params={Int[A]: 3})
    graph = pl.get(str)
    res = graph.serialize()
    schema = json_schema()
    jsonschema.validate(res, schema)
