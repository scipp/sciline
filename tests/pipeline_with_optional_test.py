# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Optional

import pytest

import sciline as sl


def test_can_provide_optional():
    def make_optional() -> Optional[int]:
        return 3

    pipeline = sl.Pipeline([make_optional])
    assert pipeline.compute(Optional[int]) == 3


def test_can_provide_optional_returning_none():
    def make_optional() -> Optional[int]:
        return None

    pipeline = sl.Pipeline([make_optional])
    assert pipeline.compute(Optional[int]) is None


def test_optional_cannot_provide_underlying():
    def make_optional() -> Optional[int]:
        return 3

    def use_int(i: int) -> float:
        return float(i)

    pipeline = sl.Pipeline([make_optional, use_int])
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.get(float)


def test_optional_dependency_can_be_filled_by_non_optional_param():
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional], params={int: 1})
    assert pipeline.compute(str) == '1'


def test_optional_dependency_is_set_to_none_if_no_provider_found():
    def use_optional(x: Optional[int]) -> str:
        return f'{x or 123}'

    pipeline = sl.Pipeline([use_optional])
    assert pipeline.compute(str) == '123'
