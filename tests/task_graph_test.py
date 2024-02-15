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


class Str(sl.Scope[T, str], str):
    ...


def to_string(x: T) -> Str[T]:
    return Str[T](str(x))


def repeat(a: A, s: Str[B]) -> list[str]:
    return [s] * a


def as_float(x: int) -> float:
    return 0.5 * x


def make_task_graph() -> Graph:
    pl = sl.Pipeline([as_float], params={int: 1})
    return pl.build(float, handler=sl.HandleAsBuildTimeException())


def test_default_scheduler_is_dask_when_dask_available() -> None:
    _ = pytest.importorskip("dask")
    tg = TaskGraph(graph={}, targets=())
    assert isinstance(tg._scheduler, sl.scheduler.DaskScheduler)


def test_compute_returns_value_when_initialized_with_single_key() -> None:
    graph = make_task_graph()
    tg = TaskGraph(graph=graph, targets=float)
    assert tg.compute() == 0.5


def test_compute_returns_dict_when_initialized_with_key_tuple() -> None:
    graph = make_task_graph()
    assert TaskGraph(graph=graph, targets=(float,)).compute() == {float: 0.5}
    assert TaskGraph(graph=graph, targets=(float, int)).compute() == {
        float: 0.5,
        int: 1,
    }


def test_compute_returns_value_when_provided_with_single_key() -> None:
    graph = make_task_graph()
    tg = TaskGraph(graph=graph, targets=float)
    assert tg.compute(int) == 1


def test_compute_returns_dict_when_provided_with_key_tuple() -> None:
    graph = make_task_graph()
    tg = TaskGraph(graph=graph, targets=float)
    assert tg.compute((int, float)) == {int: 1, float: 0.5}


def test_compute_raises_when_provided_with_key_not_in_graph() -> None:
    graph = make_task_graph()
    tg = TaskGraph(graph=graph, targets=float)
    with pytest.raises(KeyError):
        tg.compute(str)
    with pytest.raises(KeyError):
        tg.compute((str, float))


def test_keys_iter() -> None:
    pl = sl.Pipeline([to_string, repeat], params={A: 3, B: 4})
    tg = pl.get(list[str])
    assert len(list(tg.keys())) == 4  # there are no duplicates
    assert set(tg.keys()) == {A, B, Str[B], list[str]}
