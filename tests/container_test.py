# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import dask
import pytest

import sciline as sl

# We use dask with a single thread, to ensure that call counting below is correct.
dask.config.set(scheduler='synchronous')


def int_to_float(x: int) -> float:
    return 0.5 * x


def make_int() -> int:
    return 3


def int_float_to_str(x: int, y: float) -> str:
    return f"{x};{y}"


def test_make_container_sets_up_working_container():
    container = sl.Container([int_to_float, make_int])
    assert container.compute(float) == 1.5
    assert container.compute(int) == 3


def test_make_container_does_not_autobind():
    container = sl.Container([int_to_float])
    with pytest.raises(sl.UnsatisfiedRequirement):
        container.compute(float)


def test_intermediate_computed_once():
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    container = sl.Container([int_to_float, provide_int, int_float_to_str])
    assert container.compute(str) == "3;1.5"
    assert ncall == 1


def test_get_returns_task_that_computes_result():
    container = sl.Container([int_to_float, make_int])
    task = container.get(float)
    assert hasattr(task, 'compute')
    assert task.compute() == 1.5


def test_multiple_get_calls_can_be_computed_without_repeated_calls():
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    container = sl.Container([int_to_float, provide_int, int_float_to_str])
    task1 = container.get(float)
    task2 = container.get(str)
    assert dask.compute(task1, task2) == (1.5, '3;1.5')
    assert ncall == 1


def test_make_container_with_subgraph_template():
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    Param = TypeVar('Param')

    class Float(sl.Scope[Param], float):
        ...

    class Str(sl.Scope[Param], str):
        ...

    def int_float_to_str(x: int, y: Float[Param]) -> Str[Param]:
        return Str(f"{x};{y}")

    Run1 = NewType('Run1', int)
    Run2 = NewType('Run2', int)
    Result = NewType('Result', str)

    def float1() -> Float[Run1]:
        return Float[Run1](1.5)

    def float2() -> Float[Run2]:
        return Float[Run2](2.5)

    def use_strings(s1: Str[Run1], s2: Str[Run2]) -> Result:
        return Result(f"{s1};{s2}")

    container = sl.Container(
        [provide_int, float1, float2, use_strings, int_float_to_str],
    )
    assert container.get(Result).compute() == "3;1.5;3;2.5"
    assert ncall == 1


Param = TypeVar('Param')


class Str(sl.Scope[Param], str):
    ...


def f(x: Param) -> Str[Param]:
    return Str(f'{x}')


def test_container_from_templated():
    def make_float() -> float:
        return 1.5

    def combine(x: Str[int], y: Str[float]) -> str:
        return f"{x};{y}"

    container = sl.Container([make_int, make_float, combine, f])
    assert container.compute(Str[int]) == '3'
    assert container.compute(Str[float]) == '1.5'
    assert container.compute(str) == '3;1.5'
