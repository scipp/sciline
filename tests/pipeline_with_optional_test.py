# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Optional, Union

import pytest

import sciline as sl


def test_provider_returning_optional_allowed() -> None:
    def make_optional() -> int | None:
        return 3

    pl = sl.Pipeline([make_optional])
    assert pl.compute(int | None) == 3


def test_provider_returning_union_allowed() -> None:
    def make_union() -> int | float:
        return 3

    pl = sl.Pipeline([make_union])
    assert pl.compute(int | float) == 3


def test_parameter_type_union_or_optional_allowed() -> None:
    pipeline = sl.Pipeline()
    pipeline[int | float] = 3  # type: ignore[index]
    assert pipeline.compute(int | float) == 3
    pipeline[int | None] = 4  # type: ignore[index]
    assert pipeline.compute(int | None) == 4


def test_union_requirement_not_satisfied_by_any_of_its_arguments() -> None:
    def require_union(x: int | float) -> str:  # noqa: PYI041
        return f'{x}'

    pipeline = sl.Pipeline([require_union])
    pipeline[int] = 1
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.get(str)

    pipeline = sl.Pipeline([require_union])
    pipeline[float] = 1.2
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.get(str)


def test_optional_dependency_cannot_be_filled_by_non_optional_param() -> None:
    def use_optional(x: int | None) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional], params={int: 1})
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.get(str)


def test_optional_dependency_cannot_be_filled_by_non_optional_param_kwarg() -> None:
    def use_optional(*, x: int | None) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional], params={int: 1})
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.get(str)


def test_Union_argument_order_does_not_matter() -> None:
    def use_union(x: int | float) -> str:  # noqa: PYI041
        return f'{x}'

    pipeline = sl.Pipeline([use_union])
    pipeline[int | float] = 1  # type: ignore[index]
    assert pipeline.compute(str) == '1'
    pipeline = sl.Pipeline([use_union])
    pipeline[float | int] = 1  # type: ignore[index]
    assert pipeline.compute(str) == '1'
    # Note that the above works because the hashes are the same:
    assert hash(int | float) == hash(float | int)
    assert hash(Union[int, float]) == hash(Union[float, int])  # noqa: UP007


def test_optional_dependency_cannot_be_filled_transitively() -> None:
    def use_optional(x: int | None) -> str:
        return f'{x or 123}'

    def make_int(x: float) -> int:
        return int(x)

    pipeline = sl.Pipeline([use_optional, make_int], params={float: 2.2})
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.get(str)


def test_optional_dependency_can_be_set_to_None() -> None:
    def use_optional(x: int | None) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional])
    pipeline[Optional[int]] = None  # type: ignore[index] # noqa: UP007
    assert pipeline.compute(str) == '123'
