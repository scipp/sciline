# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional, Union

import pytest

import sciline as sl


def test_provider_returning_optional_disallowed():
    def make_optional() -> Optional[int]:
        return 3

    with pytest.raises(ValueError):
        sl.Pipeline([make_optional])


def test_provider_returning_union_disallowed():
    def make_union() -> Union[int, float]:
        return 3

    with pytest.raises(ValueError):
        sl.Pipeline([make_union])


def test_parameter_type_union_or_optional_disallowed():
    pipeline = sl.Pipeline()
    with pytest.raises(ValueError):
        pipeline[Union[int, float]] = 3
    with pytest.raises(ValueError):
        pipeline[Optional[int]] = 3


def test_union_requirement_leads_to_UnsatisfiedRequirement():
    def require_union(x: Union[int, float]) -> str:
        return f'{x}'

    pipeline = sl.Pipeline([require_union])
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.compute(str)


def test_optional_dependency_can_be_filled_by_non_optional_param():
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional], params={int: 1})
    assert pipeline.compute(str) == '1'


def test_union_with_none_can_be_used_instead_of_Optional():
    def use_union1(x: Union[int, None]) -> str:
        return f'{x or 123}'

    def use_union2(x: Union[None, int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_union1], params={int: 1})
    assert pipeline.compute(str) == '1'
    pipeline = sl.Pipeline([use_union2], params={int: 1})
    assert pipeline.compute(str) == '1'


def test_optional_requested_directly_can_be_filled_by_non_optional_param():
    pipeline = sl.Pipeline([], params={int: 1})
    assert pipeline.compute(Optional[int]) == 1


def test_optional_dependency_can_be_filled_transitively():
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    def make_int(x: float) -> int:
        return int(x)

    pipeline = sl.Pipeline([use_optional, make_int], params={float: 2.2})
    assert pipeline.compute(str) == '2'


def test_optional_dependency_is_set_to_none_if_no_provider_found():
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional])
    assert pipeline.compute(str) == '123'


def test_optional_dependency_is_set_to_none_if_no_provider_found_transitively():
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    def make_int(x: float) -> int:
        return int(x)

    pipeline = sl.Pipeline([use_optional, make_int])
    assert pipeline.compute(str) == '123'


def test_optional_dependency_in_node_depending_on_param_table():
    def use_optional(x: float, y: Optional[int]) -> str:
        return f'{x} {y or 123}'

    pl = sl.Pipeline([use_optional])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(sl.Series[int, str]) == sl.Series(
        int, {0: '1.0 123', 1: '2.0 123', 2: '3.0 123'}
    )
    pl[int] = 11
    assert pl.compute(sl.Series[int, str]) == sl.Series(
        int, {0: '1.0 11', 1: '2.0 11', 2: '3.0 11'}
    )


def test_optional_dependency_can_be_filled_from_param_table():
    def use_optional(x: Optional[float]) -> str:
        return f'{x or 4.0}'

    pl = sl.Pipeline([use_optional])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(sl.Series[int, str]) == sl.Series(
        int, {0: '1.0', 1: '2.0', 2: '3.0'}
    )


def test_optional_dependency_without_anchoring_param_raises_when_requesting_series():
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
