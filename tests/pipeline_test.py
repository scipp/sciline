# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import functools
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, NewType, TypeVar

import numpy as np
import numpy.typing as npt
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
    assert pipeline.compute((float, str)) == {float: 1.5, str: "3;1.5"}
    assert ncall == 1


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
    assert pipeline.compute((float, str)) == {float: 1.5, str: "3"}
    assert ncall == 1


def test_Scope_subclass_can_be_set_as_param() -> None:
    Param = TypeVar('Param')

    class Str(sl.Scope[Param, str], str):
        ...

    pipeline = sl.Pipeline(params={Str[int]: Str[int]('1')})
    pipeline[Str[float]] = Str[float]('2.0')
    assert pipeline.compute(Str[int]) == Str[int]('1')
    assert pipeline.compute(Str[float]) == Str[float]('2.0')


def test_Scope_subclass_can_be_set_as_param_with_unbound_typevar() -> None:
    Param = TypeVar('Param')

    class Str(sl.Scope[Param, str], str):
        ...

    pipeline = sl.Pipeline()
    pipeline[Str[Param]] = Str[Param]('1')  # type: ignore[valid-type]
    assert pipeline.compute(Str[int]) == Str[int]('1')
    assert pipeline.compute(Str[float]) == Str[float]('1')


def test_ScopeTwoParam_subclass_can_be_set_as_param() -> None:
    Param1 = TypeVar('Param1')
    Param2 = TypeVar('Param2')

    class Str(sl.ScopeTwoParams[Param1, Param2, str], str):
        ...

    pipeline = sl.Pipeline(params={Str[int, float]: Str[int, float]('1')})
    pipeline[Str[float, int]] = Str[float, int]('2.0')
    assert pipeline.compute(Str[int, float]) == Str[int, float]('1')
    assert pipeline.compute(Str[float, int]) == Str[float, int]('2.0')


def test_ScopeTwoParam_subclass_can_be_set_as_param_with_unbound_typevar() -> None:
    Param1 = TypeVar('Param1')
    Param2 = TypeVar('Param2')

    class Str(sl.ScopeTwoParams[Param1, Param2, str], str):
        ...

    pipeline = sl.Pipeline()
    pipeline[Str[Param1, Param2]] = Str[Param1, Param2]('1')  # type: ignore[valid-type]
    assert pipeline.compute(Str[int, float]) == Str[int, float]('1')
    assert pipeline.compute(Str[float, int]) == Str[float, int]('1')


def test_generic_providers_produce_use_dependencies_based_on_bound_typevar() -> None:
    Param = TypeVar('Param')

    class Str(sl.Scope[Param, str], str):
        ...

    def parametrized(x: Param) -> Str[Param]:
        return Str(f'{x}')

    def make_float() -> float:
        return 1.5

    def combine(x: Str[int], y: Str[float]) -> str:
        return f"{x};{y}"

    pipeline = sl.Pipeline([make_int, make_float, combine, parametrized])
    assert pipeline.compute(Str[int]) == Str[int]('3')
    assert pipeline.compute(Str[float]) == Str[float]('1.5')
    assert pipeline.compute(str) == '3;1.5'


def test_can_compute_result_depending_on_two_instances_of_generic_provider() -> None:
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    Param = TypeVar('Param')

    class Float(sl.Scope[Param, float], float):
        ...

    class Str(sl.Scope[Param, str], str):
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


def test_subclasses_of_generic_provider_defined_with_Scope_work() -> None:
    Param = TypeVar('Param')

    class StrT(sl.Scope[Param, str], str):
        ...

    class Str1(StrT[Param]):
        ...

    class Str2(StrT[Param]):
        ...

    class Str3(StrT[Param]):
        ...

    class Str4(Str3[Param]):
        ...

    def make_str1() -> Str1[Param]:
        return Str1('1')

    def make_str2() -> Str2[Param]:
        return Str2('2')

    # Note that mypy cannot detect if when setting params, the type of the
    # parameter does not match the key. Same problem as with NewType.
    pipeline = sl.Pipeline(
        [make_str1, make_str2],
        params={
            Str3[int]: Str3[int]('int3'),
            Str3[float]: Str3[float]('float3'),
            Str4[int]: Str2[int]('int4'),
        },
    )
    assert pipeline.compute(Str1[float]) == Str1[float]('1')
    assert pipeline.compute(Str2[float]) == Str2[float]('2')
    assert pipeline.compute(Str3[int]) == Str3[int]('int3')
    assert pipeline.compute(Str3[float]) == Str3[float]('float3')
    assert pipeline.compute(Str4[int]) == Str4[int]('int4')


def test_subclasses_of_generic_array_provider_defined_with_Scope_work() -> None:
    Param = TypeVar('Param')

    class ArrayT(sl.Scope[Param, npt.NDArray[np.int64]], npt.NDArray[np.int64]):
        ...

    class Array1(ArrayT[Param]):
        ...

    class Array2(ArrayT[Param]):
        ...

    def make_array1() -> Array1[Param]:
        return Array1(np.array([1, 2, 3]))

    def make_array2() -> Array2[Param]:
        return Array2(np.array([4, 5, 6]))

    pipeline = sl.Pipeline([make_array1, make_array2])
    # Note that the param is not the dtype
    assert np.all(pipeline.compute(Array1[str]) == np.array([1, 2, 3]))
    assert np.all(pipeline.compute(Array2[str]) == np.array([4, 5, 6]))


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


def test_setitem_can_replace_param_with_param() -> None:
    pl = sl.Pipeline()
    pl[int] = 1
    pl[int] = 2
    assert pl.compute(int) == 2


def test_insert_can_replace_param_with_provider() -> None:
    def func() -> int:
        return 2

    pl = sl.Pipeline()
    pl[int] = 1
    pl.insert(func)
    assert pl.compute(int) == 2


def test_setitem_can_replace_provider_with_param() -> None:
    def func() -> int:
        return 2

    pl = sl.Pipeline()
    pl.insert(func)
    pl[int] = 1
    assert pl.compute(int) == 1


def test_insert_can_replace_provider_with_provider() -> None:
    def func1() -> int:
        return 1

    def func2() -> int:
        return 2

    pl = sl.Pipeline()
    pl.insert(func1)
    pl.insert(func2)
    assert pl.compute(int) == 2


def test_insert_can_replace_generic_provider_with_generic_provider() -> None:
    T = TypeVar('T', int, float)

    @dataclass
    class A(Generic[T]):
        value: T

    def func1(x: T) -> A[T]:
        return A[T](x)

    def func2(x: T) -> A[T]:
        return A[T](x + x)

    pl = sl.Pipeline()
    pl[int] = 1
    pl.insert(func1)
    pl.insert(func2)
    assert pl.compute(A[int]) == A[int](2)


def test_insert_can_replace_generic_param_with_generic_provider() -> None:
    T = TypeVar('T', int, float)

    @dataclass
    class A(Generic[T]):
        value: T

    def func(x: T) -> A[T]:
        return A[T](x + x)

    pl = sl.Pipeline()
    pl[int] = 1
    pl[A[T]] = A[T](1)  # type: ignore[valid-type]
    assert pl.compute(A[int]) == A[int](1)
    pl.insert(func)
    assert pl.compute(A[int]) == A[int](2)


def test_setitem_can_replace_generic_provider_with_generic_param() -> None:
    T = TypeVar('T', int, float)

    @dataclass
    class A(Generic[T]):
        value: T

    def func(x: T) -> A[T]:
        return A[T](x + x)

    pl = sl.Pipeline()
    pl[int] = 1
    pl.insert(func)
    assert pl.compute(A[int]) == A[int](2)
    pl[A[T]] = A[T](1)  # type: ignore[valid-type]
    assert pl.compute(A[int]) == A[int](1)


def test_setitem_can_replace_generic_param_with_generic_param() -> None:
    T = TypeVar('T')

    @dataclass
    class A(Generic[T]):
        value: T

    pl = sl.Pipeline()
    pl[A[T]] = A[T](1)  # type: ignore[valid-type]
    assert pl.compute(A[int]) == A[int](1)
    pl[A[T]] = A[T](2)  # type: ignore[valid-type]
    assert pl.compute(A[int]) == A[int](2)


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

    class A(sl.Scope[T, int], int):
        ...

    pl = sl.Pipeline(params={A[float]: A(1), A[str]: A(2)})
    assert pl.compute(A[float]) == A(1)
    assert pl.compute(A[str]) == A(2)


def test_building_graph_with_cycle_succeeds() -> None:
    def f(x: int) -> float:
        return float(x)

    def g(x: float) -> int:
        return int(x)

    pipeline = sl.Pipeline([f, g])
    _ = pipeline.get(int)


def test_computing_graph_with_cycle_raises_CycleError() -> None:
    def f(x: int) -> float:
        return float(x)

    def g(x: float) -> int:
        return int(x)

    pipeline = sl.Pipeline([f, g])
    with pytest.raises(sl.scheduler.CycleError):
        pipeline.compute(int)


def test_get_with_single_key_return_task_graph_that_computes_value() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int, int_float_to_str])
    task = pipeline.get(str)
    assert task.compute() == '3;1.5'


@pytest.mark.parametrize('key_type', [tuple, list, iter])
def test_get_with_key_iterable_return_task_graph_that_computes_dict_of_values(
    key_type: Callable[[Any], Any],
) -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    task = pipeline.get(key_type((float, int)))
    assert task.compute() == {float: 1.5, int: 3}


def test_task_graph_compute_can_override_single_key() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    task = pipeline.get(float)
    assert task.compute(int) == 3


def test_task_graph_compute_can_override_key_tuple() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    task = pipeline.get(float)
    assert task.compute((int, float)) == {int: 3, float: 1.5}


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


def test_compute_with_NaiveScheduler() -> None:
    pipeline = sl.Pipeline([int_to_float, make_int])
    res = pipeline.compute(float, scheduler=sl.scheduler.NaiveScheduler())
    assert res == 1.5


def test_bind_and_call_no_function() -> None:
    pipeline = sl.Pipeline([make_int])
    assert pipeline.bind_and_call(()) == ()


def test_bind_and_call_function_without_args() -> None:
    def func() -> str:
        return "func"

    pipeline = sl.Pipeline([make_int])
    assert pipeline.bind_and_call(func) == "func"


def test_bind_and_call_function_with_1_arg() -> None:
    def func(i: int) -> int:
        return i * 2

    pipeline = sl.Pipeline([make_int])
    assert pipeline.bind_and_call(func) == 6


def test_bind_and_call_function_with_2_arg2() -> None:
    def func(i: int, f: float) -> float:
        return i + f

    pipeline = sl.Pipeline([make_int, int_to_float])
    assert pipeline.bind_and_call(func) == 4.5


def test_bind_and_call_overrides_default_args() -> None:
    def func(i: int, f: float = -0.5) -> float:
        return i + f

    pipeline = sl.Pipeline([make_int, int_to_float])
    assert pipeline.bind_and_call(func) == 4.5


def test_bind_and_call_function_in_iterator() -> None:
    def func(i: int) -> int:
        return i * 2

    pipeline = sl.Pipeline([make_int])
    assert pipeline.bind_and_call(iter((func,))) == (6,)


def test_bind_and_call_dataclass_without_args() -> None:
    @dataclass
    class C:
        ...

    pipeline = sl.Pipeline([make_int])
    assert pipeline.bind_and_call(C) == C()


def test_bind_and_call_dataclass_with_1_arg() -> None:
    @dataclass
    class C:
        i: int

    pipeline = sl.Pipeline([make_int])
    assert pipeline.bind_and_call(C) == C(i=3)


def test_bind_and_call_dataclass_with_2_arg2() -> None:
    @dataclass
    class C:
        i: int
        f: float

    pipeline = sl.Pipeline([make_int, int_to_float])
    assert pipeline.bind_and_call(C) == C(i=3, f=1.5)


def test_bind_and_call_two_functions() -> None:
    def func1(i: int) -> int:
        return 2 * i

    def func2(f: float) -> float:
        return f + 1

    pipeline = sl.Pipeline([make_int, int_to_float])
    assert pipeline.bind_and_call((func1, func2)) == (6, 2.5)


def test_bind_and_call_two_functions_in_iterator() -> None:
    def func1(i: int) -> int:
        return 2 * i

    def func2(f: float) -> float:
        return f + 1

    pipeline = sl.Pipeline([make_int, int_to_float])
    assert pipeline.bind_and_call(iter((func1, func2))) == (6, 2.5)


def test_bind_and_call_function_and_dataclass() -> None:
    def func(i: int) -> int:
        return 2 * i

    @dataclass
    class C:
        i: int
        f: float

    pipeline = sl.Pipeline([make_int, int_to_float])
    assert pipeline.bind_and_call((func, C)) == (6, C(i=3, f=1.5))


def test_bind_and_call_function_without_return_annotation() -> None:
    def func(i: int):  # type: ignore[no-untyped-def]
        return 2 * i

    pipeline = sl.Pipeline([make_int])
    assert pipeline.bind_and_call(func) == 6


def test_bind_and_call_generic_function() -> None:
    T = TypeVar('T')
    A = NewType('A', int)
    B = NewType('B', int)

    class G(sl.Scope[T, int], int):
        ...

    def func(a: G[A]) -> int:
        return -4 * a

    pipeline = sl.Pipeline([], params={G[A]: 3, G[B]: 4})
    assert pipeline.bind_and_call(func) == -12


def test_bind_and_call_function_runs_at_end() -> None:
    calls = []

    def a() -> int:
        calls.append('a')
        return 2

    def b() -> float:
        calls.append('b')
        return 3.1

    def c(_i: int) -> None:
        calls.append('c')

    def d(_f: float) -> None:
        calls.append('d')

    pipeline = sl.Pipeline([a, b])
    pipeline.bind_and_call([c, d])

    assert calls.index('a') in (0, 1)
    assert calls.index('b') in (0, 1)
    assert calls.index('c') in (2, 3)
    assert calls.index('d') in (2, 3)


def test_prioritizes_specialized_provider_over_generic() -> None:
    A = NewType('A', str)
    B = NewType('B', str)
    V = TypeVar('V', A, B)

    class H(sl.Scope[V, str], str):
        pass

    def p1(x: V) -> H[V]:
        return H[V]("Generic")

    def p2(x: B) -> H[B]:
        return H[B]("Special")

    pl = sl.Pipeline([p1, p2], params={A: 'A', B: 'B'})

    assert str(pl.compute(H[A])) == "Generic"
    assert str(pl.compute(H[B])) == "Special"


def test_prioritizes_specialized_provider_over_generic_several_typevars() -> None:
    A = NewType('A', str)
    B = NewType('B', str)
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    @dataclass
    class C(Generic[T1, T2]):
        first: T1
        second: T2
        third: str

    def p1(x: T1, y: T2) -> C[T1, T2]:
        return C(x, y, 'generic')

    def p2(x: A, y: T2) -> C[A, T2]:
        return C(x, y, 'medium generic')

    def p3(x: T2, y: B) -> C[T2, B]:
        return C(x, y, 'generic medium')

    def p4(x: A, y: B) -> C[A, B]:
        return C(x, y, 'special')

    pl = sl.Pipeline([p1, p2, p3, p4], params={A: A('A'), B: B('B')})

    assert pl.compute(C[B, A]) == C('B', 'A', 'generic')
    assert pl.compute(C[A, A]) == C('A', 'A', 'medium generic')
    assert pl.compute(C[B, B]) == C('B', 'B', 'generic medium')
    assert pl.compute(C[A, B]) == C('A', 'B', 'special')


def test_prioritizes_specialized_provider_raises() -> None:
    A = NewType('A', str)
    B = NewType('B', str)
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')

    @dataclass
    class C(Generic[T1, T2]):
        first: T1
        second: T2

    def p1(x: A, y: T1) -> C[A, T1]:
        return C(x, y)

    def p2(x: T1, y: B) -> C[T1, B]:
        return C(x, y)

    pl = sl.Pipeline([p1, p2], params={A: A('A'), B: B('B')})

    with pytest.raises(sl.AmbiguousProvider):
        pl.compute(C[A, B])

    with pytest.raises(sl.UnsatisfiedRequirement):
        pl.compute(C[B, A])


def test_compute_time_handler_allows_for_building_but_not_computing() -> None:
    def func(x: int) -> float:
        return 0.5 * x

    pipeline = sl.Pipeline([func])
    graph = pipeline.get(float, handler=sl.HandleAsComputeTimeException())
    with pytest.raises(sl.UnsatisfiedRequirement):
        graph.compute()


def test_pipeline_copy_simple() -> None:
    a = sl.Pipeline([int_to_float, make_int])
    b = a.copy()
    assert b.compute(int) == 3
    assert b.compute(float) == 1.5


def test_pipeline_copy_dunder() -> None:
    a = sl.Pipeline([int_to_float, make_int])
    from copy import copy

    b = copy(a)
    assert b.compute(int) == 3
    assert b.compute(float) == 1.5


def test_pipeline_copy_with_params() -> None:
    a = sl.Pipeline([int_to_float], params={int: 99})
    b = a.copy()
    assert b.compute(int) == 99
    assert b.compute(float) == 49.5


def test_pipeline_copy_after_insert() -> None:
    a = sl.Pipeline([int_to_float])
    a.insert(make_int)
    b = a.copy()
    assert b.compute(int) == 3
    assert b.compute(float) == 1.5


def test_pipeline_copy_after_setitem() -> None:
    a = sl.Pipeline([int_to_float])
    a[int] = 99
    b = a.copy()
    assert b.compute(int) == 99
    assert b.compute(float) == 49.5


def test_copy_with_generic_providers() -> None:
    Param = TypeVar('Param')

    class Str(sl.Scope[Param, str], str):
        ...

    def parametrized(x: Param) -> Str[Param]:
        return Str(f'{x}')

    def make_float() -> float:
        return 1.5

    def combine(x: Str[int], y: Str[float]) -> str:
        return f"{x};{y}"

    a = sl.Pipeline([make_int, make_float, combine, parametrized])
    b = a.copy()
    assert b.compute(Str[int]) == Str[int]('3')
    assert b.compute(Str[float]) == Str[float]('1.5')
    assert b.compute(str) == '3;1.5'


def test_pipeline_insert_on_copy_does_not_affect_original() -> None:
    a = sl.Pipeline([int_to_float])
    with pytest.raises(sl.UnsatisfiedRequirement):
        a.compute(int)
    b = a.copy()
    b.insert(make_int)
    assert b.compute(int) == 3
    assert b.compute(float) == 1.5
    with pytest.raises(sl.UnsatisfiedRequirement):
        a.compute(int)


def test_pipeline_insert_on_original_does_not_affect_copy() -> None:
    a = sl.Pipeline([int_to_float])
    b = a.copy()
    a.insert(make_int)
    assert a.compute(int) == 3
    assert a.compute(float) == 1.5
    with pytest.raises(sl.UnsatisfiedRequirement):
        b.compute(int)


def test_pipeline_setitem_on_copy_does_not_affect_original() -> None:
    a = sl.Pipeline([int_to_float])
    with pytest.raises(sl.UnsatisfiedRequirement):
        a.compute(int)
    b = a.copy()
    b[int] = 99
    assert b.compute(int) == 99
    assert b.compute(float) == 49.5
    with pytest.raises(sl.UnsatisfiedRequirement):
        a.compute(int)


def test_pipeline_setitem_on_original_does_not_affect_copy() -> None:
    a = sl.Pipeline([int_to_float])
    b = a.copy()
    a[int] = 99
    assert a.compute(int) == 99
    assert a.compute(float) == 49.5
    with pytest.raises(sl.UnsatisfiedRequirement):
        b.compute(int)


def test_pipeline_with_generics_setitem_on_original_does_not_affect_copy() -> None:
    RunType = TypeVar('RunType')

    class RawData(sl.Scope[RunType, int], int):
        ...

    class SquaredData(sl.Scope[RunType, int], int):
        ...

    Sample = NewType('Sample', int)
    Background = NewType('Background', int)
    Result = NewType('Result', int)

    def square(x: RawData[RunType]) -> SquaredData[RunType]:
        return SquaredData[RunType](x * x)

    def process(
        sample: SquaredData[Sample], background: SquaredData[Background]
    ) -> Result:
        return Result(sample + background)

    a = sl.Pipeline(
        [square, process],
        params={
            RawData[Sample]: 5,
            RawData[Background]: 2,
        },
    )
    assert a.compute(Result) == 29
    b = a.copy()
    assert b.compute(Result) == 29
    a[RawData[Sample]] = 7
    assert a.compute(Result) == 53
    assert b.compute(Result) == 29


def test_pipeline_with_generics_setitem_on_copy_does_not_affect_original() -> None:
    RunType = TypeVar('RunType')

    class RawData(sl.Scope[RunType, int], int):
        ...

    class SquaredData(sl.Scope[RunType, int], int):
        ...

    Sample = NewType('Sample', int)
    Background = NewType('Background', int)
    Result = NewType('Result', int)

    def square(x: RawData[RunType]) -> SquaredData[RunType]:
        return SquaredData[RunType](x * x)

    def process(
        sample: SquaredData[Sample], background: SquaredData[Background]
    ) -> Result:
        return Result(sample + background)

    a = sl.Pipeline(
        [square, process],
        params={
            RawData[Sample]: 5,
            RawData[Background]: 2,
        },
    )
    assert a.compute(Result) == 29
    b = a.copy()
    assert b.compute(Result) == 29
    b[RawData[Sample]] = 7
    assert a.compute(Result) == 29
    assert b.compute(Result) == 53


def test_html_repr() -> None:
    pipeline = sl.Pipeline([make_int], params={float: 5.0})
    assert isinstance(pipeline._repr_html_(), str)


def test_pipeline_keyword_only_argument_and_param(
    scheduler: sl.scheduler.Scheduler,
) -> None:
    def fn_with_kwarg(*, y: int) -> str:
        return f'y = {y}'

    pipeline = sl.Pipeline([fn_with_kwarg], params={int: 5})
    assert pipeline.compute(str, scheduler=scheduler) == 'y = 5'


def test_pipeline_mixed_arguments(scheduler: sl.scheduler.Scheduler) -> None:
    def no_args() -> float:
        return 1.2

    def pos_only(a: float) -> list[float]:
        return [a, a]

    def kwarg_only(*, lst: list[float]) -> int:
        return len(lst)

    def mixed_args(i: int, *, lst: list[float]) -> str:
        return f'i = {i}, lst[0] = {lst[0]}'

    pipeline = sl.Pipeline([no_args, pos_only, kwarg_only, mixed_args])
    assert pipeline.compute(str, scheduler=scheduler) == 'i = 2, lst[0] = 1.2'


def test_pipeline_generic_keyword_only(scheduler: sl.scheduler.Scheduler) -> None:
    T = TypeVar('T')
    A = NewType('A', int)
    B = NewType('B', int)

    class G(sl.Scope[T, int], int):
        ...

    def func(*, a: G[A]) -> int:
        return -4 * a

    pipeline = sl.Pipeline([func], params={G[A]: 3, G[B]: 4})
    assert pipeline.compute(int, scheduler=scheduler) == -12


def test_pipeline_detect_missing_argument_typehint() -> None:
    def f(x) -> int:  # type: ignore[no-untyped-def]
        return x  # type:ignore[no-any-return]

    with pytest.raises(ValueError, match='type-hint'):
        sl.Pipeline([f])


def test_pipeline_detect_missing_return_typehint() -> None:
    def f(x: int):  # type: ignore[no-untyped-def]
        return x

    with pytest.raises(ValueError, match='type-hint'):
        sl.Pipeline([f])


def test_does_not_allow_type_argument_outside_of_constraints_flat() -> None:
    T = TypeVar('T', int, float, str)
    T2 = TypeVar('T2', int, float)

    @dataclass
    class M(Generic[T]):
        value: T

    def p1(value: T2) -> M[T2]:
        return M(value)

    pipeline = sl.Pipeline((p1,))
    pipeline[str] = 'abc'
    pipeline[int] = 123

    pipeline.get(M[int])

    with pytest.raises(sl.handler.UnsatisfiedRequirement):
        pipeline.get(M[str])


def test_does_not_allow_type_argument_outside_of_constraints_nested() -> None:
    T = TypeVar('T', int, float, str)

    @dataclass
    class M(Generic[T]):
        value: T

    S = TypeVar('S', M[int], M[float], M[str])
    S2 = TypeVar('S2', M[int], M[float])

    @dataclass
    class N(Generic[S]):
        value: S

    def p1(value: T) -> M[T]:
        return M(value)

    def p2(value: S2) -> N[S2]:
        return N(value)

    pipeline = sl.Pipeline((p1, p2))
    pipeline[str] = 'abc'
    pipeline[int] = 123

    pipeline.get(N[M[int]])

    with pytest.raises(sl.handler.UnsatisfiedRequirement):
        pipeline.get(N[M[str]])


def test_constraints_nested_multiple_typevars() -> None:
    T = TypeVar('T', int, float, str)
    T2 = TypeVar('T2', int, float)

    @dataclass
    class M(Generic[T]):
        v: T

    S = TypeVar('S', M[int], M[float], M[str])
    S2 = TypeVar('S2', M[int], M[float])

    @dataclass
    class N(Generic[S, T]):
        v1: S
        v2: T

    def p1(v: T) -> M[T]:
        return M(v)

    def p2(v1: S2, v2: T2) -> N[S2, T2]:
        return N(v1, v2)

    pipeline = sl.Pipeline((p1, p2))
    pipeline[str] = 'abc'
    pipeline[int] = 123
    pipeline[float] = 3.14

    pipeline.get(N[M[float], int])
    pipeline.get(N[M[int], int])

    with pytest.raises(sl.handler.UnsatisfiedRequirement):
        pipeline.get(N[M[int], str])
    with pytest.raises(sl.handler.UnsatisfiedRequirement):
        pipeline.get(N[M[str], float])


def test_number_of_type_vars_defines_most_specialized() -> None:
    Green = NewType('Green', str)
    Blue = NewType('Blue', str)
    Color = TypeVar('Color', Green, Blue)

    @dataclass
    class Likes(Generic[Color]):
        color: Color

    Preference = TypeVar('Preference')

    @dataclass
    class Person(Generic[Preference, Color]):
        preference: Preference
        hatcolor: Color
        provided_by: str

    def p(c: Color) -> Likes[Color]:
        return Likes(c)

    def p0(p: Preference, c: Color) -> Person[Preference, Color]:
        return Person(p, c, 'p0')

    def p1(c: Color) -> Person[Likes[Color], Color]:
        return Person(Likes(c), c, 'p1')

    def p2(p: Preference) -> Person[Preference, Green]:
        return Person(p, Green('g'), 'p2')

    pipeline = sl.Pipeline((p, p0, p1, p2))
    pipeline[Blue] = 'b'
    pipeline[Green] = 'g'

    # only provided by p0
    assert pipeline.compute(Person[Likes[Green], Blue]) == Person(
        Likes(Green('g')), Blue('b'), 'p0'
    )
    # provided by p1 and p0 but p1 is preferred because it has fewer typevars
    assert pipeline.compute(Person[Likes[Blue], Blue]) == Person(
        Likes(Blue('b')), Blue('b'), 'p1'
    )
    # provided by p2 and p0 but p2 is preferred because it has fewer typevars
    assert pipeline.compute(Person[Likes[Blue], Green]) == Person(
        Likes(Blue('b')), Green('g'), 'p2'
    )

    with pytest.raises(sl.AmbiguousProvider):
        # provided by p1 and p2 with the same number of typevars
        pipeline.get(Person[Likes[Green], Green])


def test_pipeline_with_decorated_provider() -> None:
    R = TypeVar('R')

    def deco(f: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(f)
        def impl(*args: Any, **kwargs: Any) -> R:
            return f(*args, **kwargs)

        return impl

    provider = deco(int_to_float)
    pipeline = sl.Pipeline([provider], params={int: 3})
    assert pipeline.compute(float) == 1.5


def test_pipeline_with_decorated_provider_keyword_only_arg() -> None:
    R = TypeVar('R')

    def deco(f: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(f)
        def impl(*args: Any, **kwargs: Any) -> R:
            return f(*args, **kwargs)

        return impl

    @deco
    def foo(*, k: int) -> float:
        return float(k)

    provider = deco(foo)
    pipeline = sl.Pipeline([provider], params={int: 3})
    assert pipeline.compute(float) == 3.0


R = TypeVar('R')


def wrapping_decorator(f: Callable[..., R]) -> Callable[..., R]:
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        return f(*args, **kwargs)

    return wrapper


def null_decorator(f: Callable[..., R]) -> Callable[..., R]:
    return f


@pytest.mark.parametrize('deco', (null_decorator, wrapping_decorator))
def test_pipeline_lambda_provider(deco: Callable[..., Any]) -> None:
    lam = lambda x: str(x)  # noqa: E731
    lam.__annotations__['x'] = int
    lam.__annotations__['return'] = str
    func = deco(lam)

    pipeline = sl.Pipeline([func], params={int: 3})
    assert pipeline.compute(str) == '3'


@pytest.mark.parametrize('deco', (null_decorator, wrapping_decorator))
def test_pipeline_instance_method_provider(deco: Callable[..., Any]) -> None:
    class C:
        @deco
        def to_string(self, x: int) -> str:
            return str(x)

    c = C()
    pipeline = sl.Pipeline([c.to_string], params={int: 3})
    assert pipeline.compute(str) == '3'


# Must be outside of test because of string annotation
class ClassWithSelfAnnotation:
    def to_string(self: "ClassWithSelfAnnotation", x: int) -> str:
        return str(x)


def test_pipeline_instance_method_with_self_annotation_provider() -> None:
    c = ClassWithSelfAnnotation()
    pipeline = sl.Pipeline([c.to_string], params={int: 3})
    assert pipeline.compute(str) == '3'


@pytest.mark.parametrize('deco', (null_decorator, wrapping_decorator))
def test_pipeline_class_method_provider(deco: Callable[..., Any]) -> None:
    class C:
        @classmethod
        @deco
        def to_string(cls, x: int) -> str:
            return str(x)

    pipeline = sl.Pipeline([C.to_string], params={int: 3})
    assert pipeline.compute(str) == '3'


@pytest.mark.parametrize('deco', (null_decorator, wrapping_decorator))
def test_pipeline_static_method_provider(deco: Callable[..., Any]) -> None:
    class C:
        @staticmethod
        @deco
        def to_string(x: int) -> str:
            return str(x)

    pipeline = sl.Pipeline([C.to_string], params={int: 3})
    assert pipeline.compute(str) == '3'


def test_pipeline_callable_instance_provider() -> None:
    class C:
        def __init__(self, y: int) -> None:
            self.y = y

        def __call__(self, x: int) -> str:
            return str(x + self.y)

    pipeline = sl.Pipeline([C(4)], params={int: 3})
    assert pipeline.compute(str) == "7"


def test_pipeline_class_init_provider() -> None:
    class C:
        def __init__(self, x: int) -> None:
            self.x = x

    with pytest.raises(TypeError):
        sl.Pipeline([C], params={int: 3})


def test_pipeline_class_new_provider() -> None:
    class C:
        def __new__(cls, x: int) -> str:  # type: ignore[misc]
            return str(x)

    with pytest.raises(TypeError):
        sl.Pipeline([C], params={int: 3})
