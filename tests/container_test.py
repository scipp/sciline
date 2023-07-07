# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List, NewType, Tuple

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
    container = sl.make_container([int_to_float, make_int])
    assert container.get(float) == 1.5
    assert container.get(int) == 3


def test_make_container_does_not_autobind():
    container = sl.make_container([int_to_float])
    with pytest.raises(sl.UnsatisfiedRequirement):
        container.get(float)


def test_intermediate_computed_once_when_not_lazy():
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    container = sl.make_container(
        [int_to_float, provide_int, int_float_to_str], lazy=False
    )
    assert container.get(str) == "3;1.5"
    assert ncall == 1


def test_make_container_lazy_returns_task_that_computes_result():
    container = sl.make_container([int_to_float, make_int], lazy=True)
    task = container.get(float)
    assert hasattr(task, 'compute')
    assert task.compute() == 1.5


def test_lazy_with_multiple_outputs_computes_intermediates_once():
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    container = sl.make_container(
        [int_to_float, provide_int, int_float_to_str], lazy=True
    )
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

    Float = sl.parametrized_domain_type('Float', float)
    Str = sl.parametrized_domain_type('Str', str)

    def child(tp: type) -> List[Callable]:
        def int_float_to_str(x: int, y: Float[tp]) -> Str[tp]:
            return Str[tp](f"{x};{y}")

        return [int_float_to_str]

    Run1 = NewType('Run1', int)
    Run2 = NewType('Run2', int)
    Result = NewType('Result', str)

    def float1() -> Float[Run1]:
        return Float[Run1](1.5)

    def float2() -> Float[Run2]:
        return Float[Run2](2.5)

    def use_strings(s1: Str[Run1], s2: Str[Run2]) -> Result:
        return Result(f"{s1};{s2}")

    container = sl.make_container(
        [provide_int, float1, float2, use_strings] + child(Run1) + child(Run2),
        lazy=True,
    )
    assert container.get(Result).compute() == "3;1.5;3;2.5"
    assert ncall == 1


Str = sl.parametrized_domain_type('Str', str)


def subworkflow(tp: type) -> List[Callable]:
    # Could also provide option for a list of types
    # How can a user extend this? Just make there own wrapper function?
    def f(x: tp) -> Str[tp]:
        return Str[tp](x)

    return [f]


def from_templates(
    template: Callable[[type], List[Callable]], tps: Tuple[type, ...]
) -> List[Callable]:
    import itertools

    return list(itertools.chain.from_iterable(template(tp) for tp in tps))


def test_container_from_templated():
    def make_float() -> float:
        return 1.5

    def combine(x: Str[int], y: Str[float]) -> str:
        return f"{x};{y}"

    container = sl.make_container(
        [make_int, make_float, combine] + from_templates(subworkflow, (int, float))
    )
    assert container.get(Str[int]) == '3'
    assert container.get(Str[float]) == '1.5'
    assert container.get(str) == '3;1.5'
