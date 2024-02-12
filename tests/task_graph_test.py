# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import pytest

import sciline as sl
from sciline.task_graph import TaskGraph
from sciline.typing import Graph

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
    assert res.keys() == {'directed', 'multigraph', 'nodes', 'edges'}
    assert res['directed'] is True
    assert res['multigraph'] is False

    # Use predictable node labels instead of ids to check edges.
    # This is slightly ambiguous because some labels appear multiple times.
    def node_label(id_: str) -> str:
        for n in res['nodes']:
            if n['id'] == id_:
                return n['label']

    edges = [
        {'source': node_label(edge['source']), 'target': node_label(edge['target'])}
        for edge in res['edges']
    ]
    edges = sorted(edges, key=lambda e: e['source'] + e['target'])
    assert edges == expected_serialized_edges

    # Everything except the node id must be predictable.
    nodes = sorted(res['nodes'], key=lambda n: n['label'])
    for node in nodes:
        del node['id']
    assert nodes == expected_serialized_nodes


def test_ids_are_unique() -> None:
    graph = make_complex_task_graph()
    tg = TaskGraph(graph=graph, keys=str)
    res = tg.serialize()
    node_ids = [node['id'] for node in res['nodes']]
    assert len(node_ids) == len(set(node_ids))


def test_edges_refer_to_valid_ids() -> None:
    graph = make_complex_task_graph()
    tg = TaskGraph(graph=graph, keys=str)
    res = tg.serialize()
    node_ids = [node['id'] for node in res['nodes']]
    for edge in res['edges']:
        assert edge['source'] in node_ids
        assert edge['target'] in node_ids
