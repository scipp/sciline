# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from graphlib import CycleError
from typing import Generic, List, NewType, TypeVar

import pytest

import sciline as sl


def int_to_float(x: int) -> float:
    return 0.5 * x


def make_int() -> int:
    return 3


def int_float_to_str(x: int, y: float) -> str:
    return f"{x};{y}"


def test_pipeline_with_callables_can_compute_single_results() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    assert pipeline.compute(float) == 1.5
    assert pipeline.compute(int) == 3


def test_pipeline_does_not_autobind_types_that_can_be_default_constructed() -> None:
    # `int` can be constructed without arguments (and returns 0). Make sure that
    # the pipeline does not automatically bind `int` to `0`.
    pipeline = sl.Pipeline([int_to_float])
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.compute(float)


def test_intermediate_used_multiple_times_is_computed_only_once() -> None:
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    pipeline = sl.Pipeline([int_to_float, provide_int, int_float_to_str])
    assert pipeline.compute(str) == "3;1.5"
    assert ncall == 1


def test_multiple_keys_can_be_computed_without_repeated_calls() -> None:
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    pipeline = sl.Pipeline([int_to_float, provide_int, int_float_to_str])
    assert pipeline.compute((float, str)) == (1.5, "3;1.5")


def test_multiple_keys_not_in_same_path_use_same_intermediate() -> None:
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    def func1(x: int) -> float:
        return 0.5 * x

    def func2(x: int) -> str:
        return f"{x}"

    pipeline = sl.Pipeline([provide_int, func1, func2])
    assert pipeline.compute((float, str)) == (1.5, "3")
    assert ncall == 1


def test_generic_providers_produce_use_dependencies_based_on_bound_typevar() -> None:
    Param = TypeVar('Param')

    class Str(sl.Scope[Param], str):
        ...

    def parametrized(x: Param) -> Str[Param]:
        return Str(f'{x}')

    def make_float() -> float:
        return 1.5

    def combine(x: Str[int], y: Str[float]) -> str:
        return f"{x};{y}"

    pipeline = sl.Pipeline([make_int, make_float, combine, parametrized])
    assert pipeline.compute(Str[int]) == '3'
    assert pipeline.compute(Str[float]) == '1.5'
    assert pipeline.compute(str) == '3;1.5'


def test_can_compute_result_depending_on_two_instances_of_generic_provider() -> None:
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

    pipeline = sl.Pipeline(
        [provide_int, float1, float2, use_strings, int_float_to_str],
    )
    assert pipeline.compute(Result) == "3;1.5;3;2.5"
    assert ncall == 1


def test_inserting_provider_returning_None_raises() -> None:
    def provide_none() -> None:
        return None

    with pytest.raises(ValueError):
        sl.Pipeline([provide_none])
    pipeline = sl.Pipeline([])
    with pytest.raises(ValueError):
        pipeline.insert(provide_none)


def test_inserting_provider_with_no_return_type_raises() -> None:
    def provide_none():  # type: ignore[no-untyped-def]
        return None

    with pytest.raises(ValueError):
        sl.Pipeline([provide_none])
    pipeline = sl.Pipeline([])
    with pytest.raises(ValueError):
        pipeline.insert(provide_none)


def test_TypeVar_requirement_of_provider_can_be_bound() -> None:
    T = TypeVar('T')

    def provider_int() -> int:
        return 3

    def provider(x: T) -> List[T]:
        return [x, x]

    pipeline = sl.Pipeline([provider_int, provider])
    assert pipeline.compute(List[int]) == [3, 3]


def test_TypeVar_that_cannot_be_bound_raises_UnboundTypeVar() -> None:
    T = TypeVar('T')

    def provider(_: T) -> int:
        return 1

    pipeline = sl.Pipeline([provider])
    with pytest.raises(sl.UnboundTypeVar):
        pipeline.compute(int)


def test_unsatisfiable_TypeVar_requirement_of_provider_raises() -> None:
    T = TypeVar('T')

    def provider_int() -> int:
        return 3

    def provider(x: T) -> List[T]:
        return [x, x]

    pipeline = sl.Pipeline([provider_int, provider])
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.compute(List[float])


def test_TypeVar_params_are_not_associated_unless_they_match() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    class A(Generic[T1]):
        ...

    class B(Generic[T2]):
        ...

    def source() -> A[int]:
        return A[int]()

    def not_matching(x: A[T1]) -> B[T2]:
        return B[T2]()

    def matching(x: A[T1]) -> B[T1]:
        return B[T1]()

    pipeline = sl.Pipeline([source, not_matching])
    with pytest.raises(sl.UnboundTypeVar):
        pipeline.compute(B[int])

    pipeline = sl.Pipeline([source, matching])
    pipeline.compute(B[int])


def test_multi_Generic_with_fully_bound_arguments() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    @dataclass
    class A(Generic[T1, T2]):
        first: T1
        second: T2

    def source() -> A[int, float]:
        return A[int, float](1, 2.0)

    pipeline = sl.Pipeline([source])
    assert pipeline.compute(A[int, float]) == A[int, float](1, 2.0)


def test_multi_Generic_with_partially_bound_arguments() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    @dataclass
    class A(Generic[T1, T2]):
        first: T1
        second: T2

    def source() -> float:
        return 2.0

    def partially_bound(x: T1) -> A[int, T1]:
        return A[int, T1](1, x)

    pipeline = sl.Pipeline([source, partially_bound])
    assert pipeline.compute(A[int, float]) == A[int, float](1, 2.0)


def test_multi_Generic_with_multiple_unbound() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    @dataclass
    class A(Generic[T1, T2]):
        first: T1
        second: T2

    def int_source() -> int:
        return 1

    def float_source() -> float:
        return 2.0

    def unbound(x: T1, y: T2) -> A[T1, T2]:
        return A[T1, T2](x, y)

    pipeline = sl.Pipeline([int_source, float_source, unbound])
    assert pipeline.compute(A[int, float]) == A[int, float](1, 2.0)
    assert pipeline.compute(A[float, int]) == A[float, int](2.0, 1)


def test_distinct_fully_bound_instances_yield_distinct_results() -> None:
    T1 = TypeVar('T1')

    @dataclass
    class A(Generic[T1]):
        value: T1

    def int_source() -> A[int]:
        return A[int](1)

    def float_source() -> A[float]:
        return A[float](2.0)

    pipeline = sl.Pipeline([int_source, float_source])
    assert pipeline.compute(A[int]) == A[int](1)
    assert pipeline.compute(A[float]) == A[float](2.0)


def test_distinct_partially_bound_instances_yield_distinct_results() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    @dataclass
    class A(Generic[T1, T2]):
        first: T1
        second: T2

    def str_source() -> str:
        return 'a'

    def int_source(x: T1) -> A[int, T1]:
        return A[int, T1](1, x)

    def float_source(x: T1) -> A[float, T1]:
        return A[float, T1](2.0, x)

    pipeline = sl.Pipeline([str_source, int_source, float_source])
    assert pipeline.compute(A[int, str]) == A[int, str](1, 'a')
    assert pipeline.compute(A[float, str]) == A[float, str](2.0, 'a')


def test_multiple_matching_partial_providers_raises() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    @dataclass
    class A(Generic[T1, T2]):
        first: T1
        second: T2

    def int_source() -> int:
        return 1

    def float_source() -> float:
        return 2.0

    def provider1(x: T1) -> A[int, T1]:
        return A[int, T1](1, x)

    def provider2(x: T2) -> A[T2, float]:
        return A[T2, float](x, 2.0)

    pipeline = sl.Pipeline([int_source, float_source, provider1, provider2])
    assert pipeline.compute(A[int, int]) == A[int, int](1, 1)
    assert pipeline.compute(A[float, float]) == A[float, float](2.0, 2.0)
    with pytest.raises(sl.AmbiguousProvider):
        pipeline.compute(A[int, float])


def test_TypeVar_params_track_to_multiple_sources() -> None:
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    @dataclass
    class A(Generic[T1]):
        value: T1

    @dataclass
    class B(Generic[T1]):
        value: T1

    @dataclass
    class C(Generic[T1, T2]):
        first: T1
        second: T2

    def provide_int() -> int:
        return 1

    def provide_float() -> float:
        return 2.0

    def provide_A(x: T1) -> A[T1]:
        return A[T1](x)

    # Note that it currently does not matter which TypeVar instance we use here:
    # Container tracks uses of TypeVar within a single provider, but does not carry
    # the information beyond the scope of a single call.
    def provide_B(x: T1) -> B[T1]:
        return B[T1](x)

    def provide_C(x: A[T1], y: B[T2]) -> C[T1, T2]:
        return C[T1, T2](x.value, y.value)

    pipeline = sl.Pipeline(
        [provide_int, provide_float, provide_A, provide_B, provide_C]
    )
    assert pipeline.compute(C[int, float]) == C[int, float](1, 2.0)


def test_instance_provider() -> None:
    Result = NewType('Result', float)

    def f(x: int, y: float) -> Result:
        return Result(x / y)

    pl = sl.Pipeline([f])
    pl[int] = 3
    pl[float] = 2.0
    assert pl.compute(int) == 3
    assert pl.compute(float) == 2.0
    assert pl.compute(Result) == 1.5


def test_provider_NewType_instance() -> None:
    A = NewType('A', int)
    pl = sl.Pipeline([])
    pl[A] = A(3)
    assert pl.compute(A) == 3


def test_setitem_generic_sets_up_working_subproviders() -> None:
    T = TypeVar('T')

    @dataclass
    class A(Generic[T]):
        value: T

    pl = sl.Pipeline()
    pl[A[int]] = A[int](3)
    pl[A[float]] = A[float](2.0)
    assert pl.compute(A[int]) == A[int](3)
    assert pl.compute(A[float]) == A[float](2.0)
    with pytest.raises(sl.UnsatisfiedRequirement):
        pl.compute(A[str])


def test_setitem_generic_works_without_params() -> None:
    T = TypeVar('T')

    @dataclass
    class A(Generic[T]):
        value: T

    pl = sl.Pipeline()
    pl[A] = A(3)
    assert pl.compute(A) == A(3)


def test_setitem_raises_TypeError_if_instance_does_not_match_key() -> None:
    A = NewType('A', int)
    T = TypeVar('T')

    @dataclass
    class B(Generic[T]):
        value: T

    pl = sl.Pipeline()
    with pytest.raises(TypeError):
        pl[int] = 1.0
    with pytest.raises(TypeError):
        pl[A] = 1.0
    with pytest.raises(TypeError):
        pl[B[int]] = 1.0


def test_setitem_raises_if_key_exists() -> None:
    pl = sl.Pipeline()
    pl[int] = 1
    with pytest.raises(ValueError):
        pl[int] = 2


def test_init_with_params() -> None:
    pl = sl.Pipeline(params={int: 1, float: 2.0})
    assert pl.compute(int) == 1
    assert pl.compute(float) == 2.0


def test_init_with_providers_and_params() -> None:
    def func(x: int, y: float) -> str:
        return f'{x} {y}'

    pl = sl.Pipeline(providers=[func], params={int: 1, float: 2.0})
    assert pl.compute(str) == "1 2.0"


def test_init_with_sciline_Scope_subclass_param_works() -> None:
    T = TypeVar('T')

    class A(sl.Scope[T], int):
        ...

    pl = sl.Pipeline(params={A[float]: A(1), A[str]: A(2)})
    assert pl.compute(A[float]) == A(1)
    assert pl.compute(A[str]) == A(2)


def test_building_graph_with_loop_raises_CycleError() -> None:
    def f(x: int) -> float:
        return float(x)

    def g(x: float) -> int:
        return int(x)

    pipeline = sl.Pipeline([f, g])
    with pytest.raises(CycleError):
        pipeline.build(int)


def test_get_with_single_key_return_task_graph_that_computes_value() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int, int_float_to_str])
    task = pipeline.get(str)
    assert task.compute() == '3;1.5'


def test_get_with_key_tuple_return_task_graph_that_computes_tuple_of_values() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    task = pipeline.get((float, int))
    assert task.compute() == (1.5, 3)


def test_task_graph_compute_can_override_single_key() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    task = pipeline.get(float)
    assert task.compute(int) == 3


def test_task_graph_compute_can_override_key_tuple() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    task = pipeline.get(float)
    assert task.compute((int, float)) == (3, 1.5)


def test_task_graph_compute_raises_if_override_keys_outside_graph() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    task = pipeline.get(int)
    # The pipeline knows how to compute int, but the task graph does not
    # as the task graph is fixed at this point.
    with pytest.raises(KeyError):
        task.compute(float)


def test_get_with_NaiveScheduler() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    task = pipeline.get(float, scheduler=sl.scheduler.NaiveScheduler())
    assert task.compute() == 1.5
