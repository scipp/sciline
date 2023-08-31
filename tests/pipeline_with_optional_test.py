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
