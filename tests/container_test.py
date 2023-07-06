# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Callable, List, NewType

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


def test_make_child_container_inherits_bindings_from_parent():
    container = sl.make_container([int_to_float, make_int])
    child = container.make_child_container([int_float_to_str])
    assert child.get(str) == "3;1.5"


def test_make_child_container_override_parent_binding():
    def other_int() -> int:
        return 4

    container = sl.make_container([make_int])
    child = container.make_child_container([other_int, int_to_float, int_float_to_str])
    assert child.get(str) == "4;2.0"
    assert child.get(int) == 4


def test_make_child_container_override_does_not_affect_transitive_dependency():
    def other_int() -> int:
        return 4

    container = sl.make_container([int_to_float, make_int])
    child = container.make_child_container([other_int, int_float_to_str])
    assert child.get(int) == 4
    # Float computed within parent from int=3, not from other_int=4
    assert child.get(str) == "4;1.5"


def test_make_child_container_can_provide_transitive_dependency():
    # child injector must contain copy, we cannot use parent injector
    # to provide "template" for child injector
    # should we support overriding at all?

    # We had the following example in the design document:

    # import injector
    # from typing import NewType
    #
    # BackgroundContainer = NewType('BackgroundContainer', injector.Injector)
    #
    # class BackgroundModule(injector.Module):
    #     def __init__(self, container: injector.Injector):
    #         self._container = container
    #
    #     @injector.provide
    #     def get_background_iofq(self) -> BackgroundIofQ:
    #         return BackgroundIofQ(self._container.get(IofQ))
    #
    # container = injector.Injector()  # Configure with common reduction modules
    # background = container.create_child_injector(background_config)
    # background_module = BackgroundModule(background)
    # container.binder.install(background_module)

    # However, it seems like this would not work? We need to create the child injector
    # with all the relevant function, the parent may only provide inputs to those funcs.
    # Should `Container` support a registry of templates that can be used in children?
    # Or do we require manual handling?

    container = sl.make_container([int_to_float])
    child = container.make_child_container([make_int])
    assert child.get(float) == 1.5


def test_make_container_with_callable_that_uses_child():
    parent = sl.make_container([int_to_float, make_int])
    child = parent.make_child_container([int_float_to_str])

    MyStr = NewType('MyStr', str)

    def use_child() -> MyStr:
        return MyStr(child.get(str))

    container = sl.make_container([use_child])

    assert container.get(MyStr) == "3;1.5"


def test_make_container_with_multiple_children():
    parent = sl.make_container([make_int])

    def float1() -> float:
        return 1.5

    def float2() -> float:
        return 2.5

    child1 = parent.make_child_container([float1, int_float_to_str])
    child2 = parent.make_child_container([float2, int_float_to_str])
    Str1 = NewType('Str1', str)
    Str2 = NewType('Str2', str)

    def get_str1() -> Str1:
        return Str1(child1.get(str))

    def get_str2() -> Str2:
        return Str2(child2.get(str))

    def use_strings(s1: Str1, s2: Str2) -> str:
        return f"{s1};{s2}"

    container = sl.make_container([get_str1, get_str2, use_strings])
    assert container.get(Str1) == "3;1.5"
    assert container.get(Str2) == "3;2.5"
    assert container.get(str) == "3;1.5;3;2.5"


def test_make_container_with_multiple_children_does_not_repeat_calls():
    ncall = 0

    def provide_int() -> int:
        nonlocal ncall
        ncall += 1
        return 3

    parent = sl.make_container([provide_int], lazy=True)

    def float1() -> float:
        return 1.5

    def float2() -> float:
        return 2.5

    child1 = parent.make_child_container([float1, int_float_to_str])
    child2 = parent.make_child_container([float2, int_float_to_str])
    Str1 = NewType('Str1', str)
    Str2 = NewType('Str2', str)

    def get_str1() -> Str1:
        # Would need to call compute() here, but then we would not get a single graph
        # Otherwise we delay a Delayed
        # If we use lazy=False, then this computes too early.
        return Str1(child1.get(str))

    def get_str2() -> Str2:
        # Only works by coincidence, since Str2(Delayed) works. If we used
        # a function that does not take a Delayed, this would fail, e.g., sc.sin.
        return Str2(child2.get(str))

    def use_strings(s1: Str1, s2: Str2) -> str:
        return f"{s1};{s2}"

    container = sl.make_container([use_strings], lazy=True)
    # If we wrap with _injectable, things won't work. Need special handling. How?
    container._injector.binder.bind(Str1, get_str1)
    container._injector.binder.bind(Str2, get_str2)
    assert container.get(str).compute() == "3;1.5;3;2.5"
    assert ncall == 1


Str = sl.parametrized_domain_type('Str', str)


def templated(tp: type) -> List[Callable]:
    # Could also provide option for a list of types
    # How can a user extend this? Just make there own wrapper function?
    def f(x: tp) -> Str[tp]:
        return Str[tp](x)

    return [f]


def test_container_from_templated():
    def make_float() -> float:
        return 1.5

    def combine(x: Str[int], y: Str[float]) -> str:
        return f"{x};{y}"

    container = sl.make_container(
        [make_int, make_float, combine] + templated(int) + templated(float)
    )
    assert container.get(Str[int]) == '3'
    assert container.get(Str[float]) == '1.5'
    assert container.get(str) == '3;1.5'
