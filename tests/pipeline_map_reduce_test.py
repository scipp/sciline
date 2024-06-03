# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import pytest

import sciline as sl

A = NewType('A', int)
B = NewType('B', int)
C = NewType('C', int)
D = NewType('D', int)
X = NewType('X', int)


def a_to_b(a: A) -> B:
    return B(a + 1)


def b_to_c(b: B) -> C:
    return C(b + 2)


def c_to_d(c: C) -> D:
    return D(c + 4)


def test_map_returns_pipeline_that_can_compute_for_each_value() -> None:
    ab = sl.Pipeline((a_to_b,))
    mapped = ab.map({A: [A(10 * i) for i in range(3)]})
    with pytest.raises(sl.UnsatisfiedRequirement):
        # B is not in the graph any more, since it has been duplicated
        mapped.compute(B)
    from cyclebane.graph import IndexValues, NodeName

    for i in range(3):
        index = IndexValues(('dim_0',), (i,))
        assert mapped.compute(NodeName(A, index)) == A(10 * i)  # type: ignore[call-overload]
        assert mapped.compute(NodeName(B, index)) == B(10 * i + 1)  # type: ignore[call-overload]


def test_reduce_returns_pipeline_passing_mapped_branches_to_reducing_func() -> None:
    ab = sl.Pipeline((a_to_b,))
    mapped = ab.map({A: [A(10 * i) for i in range(3)]})
    # Define key to make mypy happy. This can actually be anything but currently
    # the type-hinting of Pipeline is too specific, disallowing, e.g., strings.
    Result = NewType('Result', int)
    assert mapped.reduce(func=min, name=Result).compute(Result) == Result(1)
    assert mapped.reduce(func=max, name=Result).compute(Result) == Result(21)
