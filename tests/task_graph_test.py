# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import pytest

import sciline as sl
from sciline.task_graph import TaskGraph
from sciline.typing import Graph, Json

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


def make_task_graph() -> Graph:
    pl = sl.Pipeline([as_float], params={int: 1})
    return pl.build(float, handler=sl.HandleAsBuildTimeException())


def make_complex_task_graph() -> Graph:
    pl = sl.Pipeline([make_int_b, zeros, to_string], params={Int[A]: 3})
    return pl.build(str, handler=sl.HandleAsBuildTimeException())


def test_default_scheduler_is_dask_when_dask_available() -> None:
    _ = pytest.importorskip("dask")
    tg = TaskGraph(graph={}, keys=())
    assert isinstance(tg._scheduler, sl.scheduler.DaskScheduler)


def test_compute_returns_value_when_initialized_with_single_key() -> None:
    graph = make_task_graph()
    tg = TaskGraph(graph=graph, keys=float)
    assert tg.compute() == 0.5


def test_compute_returns_dict_when_initialized_with_key_tuple() -> None:
    graph = make_task_graph()
    assert TaskGraph(graph=graph, keys=(float,)).compute() == {float: 0.5}
    assert TaskGraph(graph=graph, keys=(float, int)).compute() == {float: 0.5, int: 1}


def test_compute_returns_value_when_provided_with_single_key() -> None:
    graph = make_task_graph()
    tg = TaskGraph(graph=graph, keys=float)
    assert tg.compute(int) == 1


def test_compute_returns_dict_when_provided_with_key_tuple() -> None:
    graph = make_task_graph()
    tg = TaskGraph(graph=graph, keys=float)
    assert tg.compute((int, float)) == {int: 1, float: 0.5}


def test_compute_raises_when_provided_with_key_not_in_graph() -> None:
    graph = make_task_graph()
    tg = TaskGraph(graph=graph, keys=float)
    with pytest.raises(KeyError):
        tg.compute(str)
    with pytest.raises(KeyError):
        tg.compute((str, float))


def test_nodes_iter() -> None:
    graph = make_complex_task_graph()
    tg = TaskGraph(graph=graph, keys=str)
    keys = set(n for n in tg.nodes() if not isinstance(n, sl.Provider))
    functions = set(
        n.func
        for n in tg.nodes()
        if isinstance(n, sl.Provider) and n.kind == 'function'
    )
    params = set(
        n.call({})
        for n in tg.nodes()
        if isinstance(n, sl.Provider) and n.kind == 'parameter'
    )
    # We got all nodes when splitting sets
    # -1 because `zeros` shows up twice in `nodes()` but not `functions`
    assert len(keys) + len(functions) + len(params) == len(list(tg.nodes())) - 1

    assert keys == {
        Int[A],
        Int[B],
        List[A],
        List[B],
        str,
    }
    assert functions == {make_int_b, zeros, to_string}
    assert params == {3}


def test_edges_iter() -> None:
    graph = make_complex_task_graph()
    tg = TaskGraph(graph=graph, keys=str)
    from_function = set(
        (f.func, t)
        for (f, t) in tg.edges()
        if isinstance(f, sl.Provider)
        and not isinstance(t, sl.Provider)
        and f.kind == 'function'
    )
    from_param = set(
        (f.call({}), t)
        for (f, t) in tg.edges()
        if isinstance(f, sl.Provider)
        and not isinstance(t, sl.Provider)
        and f.kind == 'parameter'
    )
    to_provider = set(
        (f, t.func)
        for (f, t) in tg.edges()
        if not isinstance(f, sl.Provider) and isinstance(t, sl.Provider)
    )
    # We got all nodes when splitting sets
    assert len(from_function) + len(from_param) + len(to_provider) == len(
        list(tg.edges())
    )
    assert from_function == {
        (make_int_b, Int[B]),
        (zeros, List[A]),
        (zeros, List[B]),
        (to_string, str),
    }
    assert from_param == {(3, Int[A])}
    assert to_provider == {
        (Int[A], zeros),
        (Int[B], zeros),
        (List[A], to_string),
        (List[B], to_string),
    }


def check_serialized_graph(serialized: Json, expected_nodes, expected_edges) -> None:
    assert serialized.keys() == {'directed', 'multigraph', 'nodes', 'edges'}
    assert serialized['directed'] is True
    assert serialized['multigraph'] is False

    # Use predictable node labels instead of ids to check edges.
    # This is slightly ambiguous because some labels appear multiple times.
    def node_label(id_: str) -> str:
        for n in serialized['nodes']:
            if n['id'] == id_:
                return n['label']

    edges = [
        {'source': node_label(edge['source']), 'target': node_label(edge['target'])}
        for edge in serialized['edges']
    ]
    edges = sorted(edges, key=lambda e: e['source'] + e['target'])
    assert edges == expected_edges

    # Everything except the node id must be predictable.
    nodes = sorted(serialized['nodes'], key=lambda n: n['label'])
    for node in nodes:
        del node['id']
    assert nodes == expected_nodes


# Result of serializing make_complex_task_graph(), sorted by label.
expected_serialized_nodes = [
    {
        'label': 'Int[A]',
        'kind': 'data',
        'type': 'task_graph_test.Int[task_graph_test.A]',
    },
    {
        'label': 'Int[A]',
        'kind': 'p_parameter',
        'type': 'task_graph_test.Int[task_graph_test.A]',
    },
    {
        'label': 'Int[B]',
        'kind': 'data',
        'type': 'task_graph_test.Int[task_graph_test.B]',
    },
    {
        'label': 'List[A]',
        'kind': 'data',
        'type': 'task_graph_test.List[task_graph_test.A]',
    },
    {
        'label': 'List[B]',
        'kind': 'data',
        'type': 'task_graph_test.List[task_graph_test.B]',
    },
    {
        'label': 'make_int_b',
        'kind': 'p_function',
        'function': 'task_graph_test.make_int_b',
    },
    {'label': 'str', 'kind': 'data', 'type': 'builtins.str'},
    {
        'label': 'to_string',
        'kind': 'p_function',
        'function': 'task_graph_test.to_string',
    },
    {'label': 'zeros', 'kind': 'p_function', 'function': 'task_graph_test.zeros'},
    {'label': 'zeros', 'kind': 'p_function', 'function': 'task_graph_test.zeros'},
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


def test_serialize() -> None:
    # We cannot easily test the graph structure because we cannot predict node ids.
    graph = make_complex_task_graph()
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
        'function': 'task_graph_test.as_float',
    },
    {
        'label': 'as_float',
        'kind': 'p_function',
        'function': 'task_graph_test.as_float',
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
    graph = make_complex_task_graph()
    tg = TaskGraph(graph=graph, keys=str)
    res = tg.serialize()
    node_ids = [node['id'] for node in res['nodes']]
    assert len(node_ids) == len(set(node_ids))
