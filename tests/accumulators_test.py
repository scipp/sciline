# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable

import pytest

from sciline.accumulators import Accumulator, Buffered, Reduced


def add(*parts: float) -> float:
    return sum(parts)


def test_buffered_applies_func_in_push_order() -> None:
    def join(*parts: str) -> str:
        return ''.join(parts)

    acc = Buffered(join)
    acc.push('x')
    acc.push('y')
    acc.push('z')
    assert acc.value == 'xyz'


def test_buffered_without_pushes_has_no_value() -> None:
    acc = Buffered(add)
    with pytest.raises(ValueError, match='Nothing has been pushed'):
        acc.value


def test_reduced_applies_func_in_push_order() -> None:
    def join(left: str, right: str) -> str:
        return left + right

    acc = Reduced(join)
    acc.push('x')
    acc.push('y')
    acc.push('z')
    assert acc.value == 'xyz'


def test_reduced_without_pushes_has_no_value() -> None:
    acc = Reduced(add)
    with pytest.raises(ValueError, match='Nothing has been pushed'):
        acc.value


def test_reduced_takes_the_first_push_as_result() -> None:
    pushed: list[list[float]] = []

    def merge(left: list[float], right: list[float]) -> list[float]:
        pushed.append(right)
        return left + right

    acc = Reduced(merge)
    for part in ([1.0], [2.0], [3.0]):
        acc.push(part)
    assert acc.value == [1.0, 2.0, 3.0]
    assert pushed == [[2.0], [3.0]]


@pytest.mark.parametrize('make', [lambda: Buffered(add), lambda: Reduced(add)])
def test_pushing_combined_values_gives_the_same_result(
    make: Callable[[], Accumulator[float]],
) -> None:
    whole = make()
    groups = [make(), make()]
    for i, x in enumerate([1.0, 2.0, 3.0, 4.0]):
        whole.push(x)
        groups[i % 2].push(x)
    chained = make()
    for g in groups:
        chained.push(g.value)
    assert chained.value == whole.value
