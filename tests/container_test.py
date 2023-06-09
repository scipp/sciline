# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import dask
import pytest

import sciline as sl

# We use dask with a single thread, to ensure that call counting below is correct.
dask.config.set(scheduler='synchronous')


def f(x: int) -> float:
    return 0.5 * x


def g() -> int:
    return 3


def h(x: int, y: float) -> str:
    return f"{x};{y}"


def test_make_container_sets_up_working_container():
    container = sl.make_container([f, g])
    assert container.get(float) == 1.5
    assert container.get(int) == 3


def test_make_container_does_not_autobind():
    container = sl.make_container([f])
    with pytest.raises(sl.UnsatisfiedRequirement):
        container.get(float)


def test_intermediate_computed_once_when_not_lazy():
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    container = sl.make_container([f, provide_int, h], lazy=False)
    assert container.get(str) == "3;1.5"
    assert ncall == 1


def test_make_container_lazy_returns_task_that_computes_result():
    container = sl.make_container([f, g], lazy=True)
    task = container.get(float)
    assert hasattr(task, 'compute')
    assert task.compute() == 1.5


def test_lazy_with_multiple_outputs_computes_intermediates_once():
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    container = sl.make_container([f, provide_int, h], lazy=True)
    task1 = container.get(float)
    task2 = container.get(str)
    assert dask.compute(task1, task2) == (1.5, '3;1.5')
    assert ncall == 1
