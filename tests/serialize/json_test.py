# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sys
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


def to_string(a: List[A], b: List[B]) -> str:
    return f'a: {a}, b: {b}'


def as_float(x: int) -> float:
    return 0.5 * x


def check_serialized_graph(
    serialized: dict[str, Json], expected_nodes: Any, expected_edges: Any
) -> None:
    assert serialized.keys() == {'directed', 'multigraph', 'nodes', 'edges'}
    assert serialized['directed'] is True
    assert serialized['multigraph'] is False

    # Use predictable node labels instead of ids to check edges.
    # This is slightly ambiguous because some labels appear multiple times.
    def node_label(id_: str) -> str:
        for n in serialized['nodes']:
            if n['id'] == id_:  # type: ignore[index]
                return n['label']  # type: ignore[index, return-value]
        raise AssertionError("Edge refers to invalid node id")

    edges = [
        {
            'source': node_label(edge['source']),  # type: ignore[arg-type]
            'target': node_label(edge['target']),  # type: ignore[arg-type]
        }
        for edge in serialized['edges']
    ]
    edges = sorted(edges, key=lambda e: e['source'] + e['target'])
    assert edges == expected_edges

    # Everything except the node id must be predictable.
    nodes = sorted(
        serialized['nodes'], key=lambda n: str(n['label'])  # type: ignore[arg-type]
    )
    for node in nodes:
        del node['id']  # type: ignore[arg-type]
    assert nodes == expected_nodes


# Result of serializing make_complex_task_graph(), sorted by label.
expected_serialized_nodes = [
    {
        'label': 'Int[A]',
        'kind': 'data',
        'type': 'json_test.Int[json_test.A]',
    },
    {
        'label': 'Int[A]',
        'kind': 'p_parameter',
        'type': 'json_test.Int[json_test.A]',
    },
    {
        'label': 'Int[B]',
        'kind': 'data',
        'type': 'json_test.Int[json_test.B]',
    },
    {
        'label': 'List[A]',
        'kind': 'data',
        'type': 'json_test.List[json_test.A]',
    },
    {
        'label': 'List[B]',
        'kind': 'data',
        'type': 'json_test.List[json_test.B]',
    },
    {
        'label': 'make_int_b',
        'kind': 'p_function',
        'function': 'json_test.make_int_b',
    },
    {'label': 'str', 'kind': 'data', 'type': 'builtins.str'},
    {
        'label': 'to_string',
        'kind': 'p_function',
        'function': 'json_test.to_string',
    },
    {'label': 'zeros', 'kind': 'p_function', 'function': 'json_test.zeros'},
    {'label': 'zeros', 'kind': 'p_function', 'function': 'json_test.zeros'},
]
# Ids were replaced by labels here.
expected_serialized_edges = [
    {'source': 'Int[A]', 'target': 'Int[A]'},
    {'source': 'Int[A]', 'target': 'zeros'},
    {'source': 'Int[B]', 'target': 'zeros'},
    {'source': 'List[A]', 'target': 'to_string'},
    {'source': 'List[B]', 'target': 'to_string'},
    {'source': 'make_int_b', 'target': 'Int[B]'},
    {'source': 'to_string', 'target': 'str'},
    {'source': 'zeros', 'target': 'List[A]'},
    {'source': 'zeros', 'target': 'List[B]'},
]


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_serialize() -> None:
    # We cannot easily test the graph structure because we cannot predict node ids.
    pl = sl.Pipeline([make_int_b, zeros, to_string], params={Int[A]: 3})
    graph = pl.build(str, handler=sl.HandleAsBuildTimeException())
    tg = TaskGraph(graph=graph, keys=str)
    res = tg.serialize()
    check_serialized_graph(res, expected_serialized_nodes, expected_serialized_edges)


# Result of serializing a task graph with a param table, sorted by label.
expected_serialized_nodes_param_table = [
    {
        'label': 'Series[str, float]',
        'kind': 'data',
        'type': 'sciline.series.Series[builtins.str, builtins.float]',
    },
    {
        'label': 'as_float',
        'kind': 'p_function',
        'function': 'json_test.as_float',
    },
    {
        'label': 'as_float',
        'kind': 'p_function',
        'function': 'json_test.as_float',
    },
    {
        'label': 'float(str:0)',
        'kind': 'data_table_cell',
        'value_type': 'builtins.float',
        'row_types': ['builtins.str'],
        'row_indices': ['0'],
    },
    {
        'label': 'float(str:1)',
        'kind': 'data_table_cell',
        'value_type': 'builtins.float',
        'row_types': ['builtins.str'],
        'row_indices': ['1'],
    },
    {
        'label': 'int(str:0)',
        'kind': 'data_table_cell',
        'value_type': 'builtins.int',
        'row_types': ['builtins.str'],
        'row_indices': ['0'],
    },
    {
        'label': 'int(str:1)',
        'kind': 'data_table_cell',
        'value_type': 'builtins.int',
        'row_types': ['builtins.str'],
        'row_indices': ['1'],
    },
    {
        'label': 'provide_series[str, float]',
        'kind': 'p_series',
        'value_type': 'builtins.float',
        'row_dim': 'builtins.str',
        'labels': ['0', '1'],
    },
    {
        'label': 'table_cell(int(str:0))',
        'kind': 'p_table_cell',
    },
    {
        'label': 'table_cell(int(str:1))',
        'kind': 'p_table_cell',
    },
]
expected_serialized_edges_param_table = [
    {'source': 'as_float', 'target': 'float(str:0)'},
    {'source': 'as_float', 'target': 'float(str:1)'},
    {'source': 'float(str:0)', 'target': 'provide_series[str, float]'},
    {'source': 'float(str:1)', 'target': 'provide_series[str, float]'},
    {'source': 'int(str:0)', 'target': 'as_float'},
    {'source': 'int(str:1)', 'target': 'as_float'},
    {'source': 'provide_series[str, float]', 'target': 'Series[str, float]'},
    {'source': 'table_cell(int(str:0))', 'target': 'int(str:0)'},
    {'source': 'table_cell(int(str:1))', 'target': 'int(str:1)'},
]


def test_serialize_param_table() -> None:
    pl = sl.Pipeline([as_float])
    pl.set_param_table(sl.ParamTable(str, {int: [3, -5]}))
    graph = pl.build(sl.Series[str, float], handler=sl.HandleAsBuildTimeException())
    tg = TaskGraph(graph=graph, keys=sl.Series[str, float])
    res = tg.serialize()
    check_serialized_graph(
        res,
        expected_serialized_nodes_param_table,
        expected_serialized_edges_param_table,
    )


def test_serialize_ids_are_unique() -> None:
    pl = sl.Pipeline([make_int_b, zeros, to_string], params={Int[A]: 3})
    graph = pl.build(str, handler=sl.HandleAsBuildTimeException())
    tg = TaskGraph(graph=graph, keys=str)
    res = tg.serialize()
    node_ids = [node['id'] for node in res['nodes']]
    assert len(node_ids) == len(set(node_ids))


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
