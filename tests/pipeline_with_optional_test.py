# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional, Union

import pytest

import sciline as sl


def test_provider_returning_optional_disallowed() -> None:
    def make_optional() -> Optional[int]:
        return 3

    with pytest.raises(ValueError):
        sl.Pipeline([make_optional])


def test_provider_returning_union_disallowed() -> None:
    def make_union() -> Union[int, float]:
        return 3

    with pytest.raises(ValueError):
        sl.Pipeline([make_union])


def test_parameter_type_union_or_optional_disallowed() -> None:
    pipeline = sl.Pipeline()
    with pytest.raises(ValueError):
        pipeline[Union[int, float]] = 3  # type: ignore[index]
    with pytest.raises(ValueError):
        pipeline[Optional[int]] = 3  # type: ignore[index]


def test_union_requirement_satisfied_by_unique_match() -> None:
    def require_union(x: Union[int, float]) -> str:
        return f'{x}'

    pipeline = sl.Pipeline([require_union])
    pipeline[int] = 1
    assert pipeline.compute(str) == '1'


def test_union_requirement_with_multiple_matches_raises_AmbiguousProvider() -> None:
    def require_union(x: Union[int, float]) -> str:
        return f'{x}'

    pipeline = sl.Pipeline([require_union])
    pipeline[int] = 1
    pipeline[float] = 1.5
    with pytest.raises(sl.AmbiguousProvider):
        pipeline.compute(str)


def test_optional_dependency_can_be_filled_by_non_optional_param(
    scheduler: sl.scheduler.Scheduler,
) -> None:
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional], params={int: 1})
    assert pipeline.compute(str, scheduler=scheduler) == '1'


def test_optional_dependency_can_be_filled_by_non_optional_param_kwarg(
    scheduler: sl.scheduler.Scheduler,
) -> None:
    def use_optional(*, x: Optional[int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional], params={int: 1})
    assert pipeline.compute(str, scheduler=scheduler) == '1'


def test_union_with_none_can_be_used_instead_of_Optional() -> None:
    def use_union1(x: Union[int, None]) -> str:
        return f'{x or 123}'

    def use_union2(x: Union[None, int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_union1], params={int: 1})
    assert pipeline.compute(str) == '1'
    pipeline = sl.Pipeline([use_union2], params={int: 1})
    assert pipeline.compute(str) == '1'


def test_optional_requested_directly_can_be_filled_by_non_optional_param() -> None:
    pipeline = sl.Pipeline([], params={int: 1})
    assert pipeline.compute(Optional[int]) == 1  # type: ignore[call-overload]


def test_optional_dependency_can_be_filled_transitively() -> None:
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    def make_int(x: float) -> int:
        return int(x)

    pipeline = sl.Pipeline([use_optional, make_int], params={float: 2.2})
    assert pipeline.compute(str) == '2'


def test_optional_dependency_is_set_to_none_if_no_provider_found(
    scheduler: sl.scheduler.Scheduler,
) -> None:
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional])
    assert pipeline.compute(str, scheduler=scheduler) == '123'


def test_optional_dependency_is_set_to_none_if_no_provider_found_kwarg(
    scheduler: sl.scheduler.Scheduler,
) -> None:
    def use_optional(*, x: Optional[int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional])
    assert pipeline.compute(str, scheduler=scheduler) == '123'


def test_optional_dependency_is_set_to_none_if_no_provider_found_transitively() -> None:
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    def make_int(x: float) -> int:
        return int(x)

    pipeline = sl.Pipeline([use_optional, make_int])
    assert pipeline.compute(str) == '123'


def test_can_have_both_optional_and_non_optional_path_to_param() -> None:
    Str1 = NewType('Str1', str)
    Str2 = NewType('Str2', str)
    Str12 = NewType('Str12', str)
    Str21 = NewType('Str21', str)

    def use_optional_int(x: Optional[int]) -> Str1:
        return Str1(f'{x or 123}')

    def use_int(x: int) -> Str2:
        return Str2(f'{x}')

    def combine12(x: Str1, y: Str2) -> Str12:
        return Str12(f'{x} {y}')

    def combine21(x: Str2, y: Str1) -> Str21:
        return Str21(f'{x} {y}')

    pipeline = sl.Pipeline(
        [use_optional_int, use_int, combine12, combine21], params={int: 1}
    )
    assert pipeline.compute(Str12) == '1 1'
    assert pipeline.compute(Str21) == '1 1'


def test_presence_of_optional_does_not_affect_related_exception() -> None:
    Str1 = NewType('Str1', str)
    Str2 = NewType('Str2', str)
    Str12 = NewType('Str12', str)
    Str21 = NewType('Str21', str)

    def use_optional_int(x: Optional[int]) -> Str1:
        return Str1(f'{x or 123}')

    # Make sure the implementation does not unintentionally put "None" here,
    # triggered by the presence of the optional dependency on int in another provider.
    def use_int(x: int) -> Str2:
        return Str2(f'{x}')

    def combine12(x: Str1, y: Str2) -> Str12:
        return Str12(f'{x} {y}')

    def combine21(x: Str2, y: Str1) -> Str21:
        return Str21(f'{x} {y}')

    pipeline = sl.Pipeline([use_optional_int, use_int, combine12, combine21])
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.compute(Str12)
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.compute(Str21)


def test_optional_dependency_in_node_depending_on_param_table() -> None:
    Int = NewType('Int', int)

    def use_optional(x: float, y: Optional[Int]) -> str:
        return f'{x} {y or 123}'

    pl = sl.Pipeline([use_optional])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(sl.Series[int, str]) == sl.Series(
        int, {0: '1.0 123', 1: '2.0 123', 2: '3.0 123'}
    )
    pl[Int] = 11
    assert pl.compute(sl.Series[int, str]) == sl.Series(
        int, {0: '1.0 11', 1: '2.0 11', 2: '3.0 11'}
    )


def test_optional_dependency_can_be_filled_from_param_table() -> None:
    def use_optional(x: Optional[float]) -> str:
        return f'{x or 4.0}'

    pl = sl.Pipeline([use_optional])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(sl.Series[int, str]) == sl.Series(
        int, {0: '1.0', 1: '2.0', 2: '3.0'}
    )


def test_optional_without_anchoring_param_raises_when_requesting_series() -> None:
    Param = NewType('Param', float)

    def use_optional(x: Optional[float]) -> str:
        return f'{x or 4.0}'

    pl = sl.Pipeline([use_optional])
    pl.set_param_table(sl.ParamTable(int, {Param: [1.0, 2.0, 3.0]}))
    # It is a bit ambiguous what we would expect here: Above, we have another param
    # used from the table, defining the length of the series. Here, we could replicate
    # the output of use_optional(None) based on the `int` param table:
    #    sl.Series(int, {0: '4.0', 1: '4.0', 2: '4.0'})
    # However, we are not supporting this for non-optional dependencies either since
    # it is unclear whether that would bring conceptual issues or risk.
    with pytest.raises(sl.UnsatisfiedRequirement):
        pl.compute(sl.Series[int, str])
