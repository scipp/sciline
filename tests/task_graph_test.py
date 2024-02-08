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


"""
{
    'directed': True,
    'multigraph': False,
    'graph': {},
    'nodes': [
        {'label': 'A', 'f': 'foo.a', 'id': 'a'},
        {'label': 'B', 'f': 'foo.b', 'id': 'b'},
        {'label': 'C', 'f': 'bar.c', 'id': 'c'}
    ],
    'links': [
        {'weight': 1.2, 'source': 'a', 'target': 'b'},
        {'weight': 2.4, 'source': 'a', 'target': 'c'}
    ]
}
"""


def test_serialize() -> None:
    # We cannot easily test the graph structure because we cannot predict node ids.
    graph = make_complex_task_graph()
    tg = TaskGraph(graph=graph, keys=str)
    res = tg.serialize()
    assert res.keys() == {'directed', 'multigraph', 'nodes', 'edges'}
    assert res['directed'] is True
    assert res['multigraph'] is False

    nodes = sorted(res['nodes'], key=lambda node: node['label'])
    for node in nodes:
        del node['id']
    assert nodes == [
        {'label': 'Int[A]', 'kind': 'data', 'type': 'task_graph_test.Int[A]'},
    ]


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
