# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# type: ignore

import sys
from copy import deepcopy
from typing import Any, NewType, TypeVar

import jsonschema
import pytest

import sciline as sl
from sciline.serialize import json_schema
from sciline.task_graph import TaskGraph
from sciline.typing import Json

A = NewType('A', int)
B = NewType('B', int)
T = TypeVar('T', A, B)


class Int(sl.Scope[T, int], int):
    ...


class List(sl.Scope[T, list[int]], list[int]):
    ...


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
    id_mapping = {}
    nodes: Any = sorted(deepcopy(graph['nodes']), key=lambda n: n['out'])
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
        'label': 'to_string',
        'kind': 'function',
        'function': 'json_test.to_string',
        'out': 'builtins.str',
        'args': ['103', '102'],
        'kwargs': {},
    },
    {
        'id': '1',
        'label': 'Int[A]',
        'kind': 'parameter',
        'out': 'json_test.Int[json_test.A]',
    },
    {
        'id': '2',
        'label': 'make_int_b',
        'kind': 'function',
        'function': 'json_test.make_int_b',
        'out': 'json_test.Int[json_test.B]',
        'args': [],
        'kwargs': {},
    },
    {
        'id': '3',
        'label': 'zeros',
        'kind': 'function',
        'function': 'json_test.zeros',
        'out': 'json_test.List[json_test.A]',
        'args': ['100'],
        'kwargs': {},
    },
    {
        'id': '4',
        'label': 'zeros',
        'kind': 'function',
        'function': 'json_test.zeros',
        'out': 'json_test.List[json_test.B]',
        'args': ['101'],
        'kwargs': {},
    },
]
expected_serialized_edges = [
    {'id': '100', 'source': '1', 'target': '3'},
    {'id': '101', 'source': '2', 'target': '4'},
    {'id': '102', 'source': '3', 'target': '0'},
    {'id': '103', 'source': '4', 'target': '0'},
]
expected_serialized_graph = {
    'directed': True,
    'multigraph': False,
    'nodes': expected_serialized_nodes,
    'edges': expected_serialized_edges,
}


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
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
        'label': 'fn_w_kwonlyargs',
        'kind': 'function',
        'function': 'json_test.fn_w_kwonlyargs',
        'out': 'builtins.float',
        'args': [],
        'kwargs': {'x': '100'},
    },
    {
        'id': '1',
        'label': 'int',
        'kind': 'parameter',
        'out': 'builtins.int',
    },
]
expected_serialized_kwonlyargs_edges = [
    {'id': '100', 'source': '1', 'target': '0'},
]
expected_serialized_kwonlyargs_graph = {
    'directed': True,
    'multigraph': False,
    'nodes': expected_serialized_kwonlyargs_nodes,
    'edges': expected_serialized_kwonlyargs_edges,
}


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_serialize_kwonlyargs() -> None:
    pl = sl.Pipeline([fn_w_kwonlyargs], params={int: 3})
    graph = pl.get(float)
    res = graph.serialize()
    res = make_graph_predictable(res)
    assert res == expected_serialized_kwonlyargs_graph


def repeated_arg(a: str, b: str) -> list[str]:
    return [a, b]


# Ids correspond to the result of assign_predictable_ids
expected_serialized_repeated_arg_nodes = [
    {
        'id': '0',
        'label': 'repeated_arg',
        'kind': 'function',
        'function': 'json_test.repeated_arg',
        'out': 'builtins.list[builtins.str]',
        'args': ['100', '101'],
        'kwargs': {},
    },
    {
        'id': '1',
        'label': 'str',
        'kind': 'parameter',
        'out': 'builtins.str',
    },
]
expected_serialized_repeated_arg_edges = [
    # The edge is repeated
    {'id': '100', 'source': '1', 'target': '0'},
    {'id': '101', 'source': '1', 'target': '0'},
]
expected_serialized_repeated_arg_graph = {
    'directed': True,
    'multigraph': False,
    'nodes': expected_serialized_repeated_arg_nodes,
    'edges': expected_serialized_repeated_arg_edges,
}


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_serialize_repeated_arg() -> None:
    pl = sl.Pipeline([repeated_arg], params={str: 'abc'})
    graph = pl.get(list[str])
    res = graph.serialize()
    res = make_graph_predictable(res)
    assert res == expected_serialized_repeated_arg_graph


def repeated_arg_konlywarg(a: str, *, b: str) -> list[str]:
    return [a, b]


def repeated_konlywargs(*, a: int, b: int) -> str:
    return str(a + b)


# Ids correspond to the result of assign_predictable_ids
expected_serialized_repeated_konlywarg_nodes = [
    {
        'id': '0',
        'label': 'int',
        'kind': 'parameter',
        'out': 'builtins.int',
    },
    {
        'id': '1',
        'label': 'repeated_arg_konlywarg',
        'kind': 'function',
        'function': 'json_test.repeated_arg_konlywarg',
        'out': 'builtins.list[builtins.str]',
        'args': ['102'],
        'kwargs': {'b': '103'},
    },
    {
        'id': '2',
        'label': 'repeated_konlywargs',
        'kind': 'function',
        'function': 'json_test.repeated_konlywargs',
        'out': 'builtins.str',
        'args': [],
        'kwargs': {'a': '100', 'b': '101'},
    },
]
expected_serialized_repeated_konlywarg_edges = [
    # The edges are repeated
    {'id': '100', 'source': '0', 'target': '2'},
    {'id': '101', 'source': '0', 'target': '2'},
    {'id': '102', 'source': '2', 'target': '1'},
    {'id': '103', 'source': '2', 'target': '1'},
]
expected_serialized_repeated_konlywarg_graph = {
    'directed': True,
    'multigraph': False,
    'nodes': expected_serialized_repeated_konlywarg_nodes,
    'edges': expected_serialized_repeated_konlywarg_edges,
}


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_serialize_repeated_konlywarg() -> None:
    pl = sl.Pipeline([repeated_arg_konlywarg, repeated_konlywargs], params={int: 4})
    graph = pl.get(list[str])
    res = graph.serialize()
    res = make_graph_predictable(res)
    assert res == expected_serialized_repeated_konlywarg_graph


def test_serialize_param_table() -> None:
    pl = sl.Pipeline([as_float])
    pl.set_param_table(sl.ParamTable(str, {int: [3, -5]}))
    graph = pl.get(sl.Series[str, float])
    with pytest.raises(ValueError):
        graph.serialize()


def test_serialize_validate_schema() -> None:
    pl = sl.Pipeline([make_int_b, zeros, to_string], params={Int[A]: 3})
    graph = pl.build(str, handler=sl.HandleAsBuildTimeException())
    tg = TaskGraph(graph=graph, keys=str)
    res = tg.serialize()
    schema = json_schema()
    jsonschema.validate(res, schema)


def test_serialize_validate_schema_param_table() -> None:
    pl = sl.Pipeline([as_float])
    pl.set_param_table(sl.ParamTable(str, {int: [3, -5]}))
    graph = pl.build(sl.Series[str, float], handler=sl.HandleAsBuildTimeException())
    tg = TaskGraph(graph=graph, keys=str)
    res = tg.serialize()
    schema = json_schema()
    jsonschema.validate(res, schema)
