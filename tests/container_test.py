# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

import sciline as sl


def test_make_container_sets_up_working_container():
    def f(x: int) -> float:
        return 0.5 * x

    def g() -> int:
        return 3

    container = sl.make_container([f, g])
    assert container.get(float) == 1.5
    assert container.get(int) == 3


def test_make_container_does_not_autobind():
    def f(x: int) -> float:
        return 0.5 * x

    container = sl.make_container([f])
    with pytest.raises(sl.UnsatisfiedRequirement):
        container.get(float)
