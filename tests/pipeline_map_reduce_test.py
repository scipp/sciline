# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import NewType

import pandas as pd
import pytest
from cyclebane.graph import IndexValues, NodeName

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


def test_compute_series_single_index() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    paramsA = pd.DataFrame(
        {A: [A(10 * i) for i in range(3)]}, index=['a', 'b', 'c']
    ).rename_axis('x')
    mapped = pl.map(paramsA)
    result = sl.compute_series(mapped, C)
    assert result['a'] == C(7)
    assert result['b'] == C(17)
    assert result['c'] == C(27)
    assert result.index.name == 'x'
    assert result.name == C


def test_compute_series_single_index_with_no_name() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    paramsA = pd.DataFrame({A: [A(10 * i) for i in range(3)]}, index=['a', 'b', 'c'])
    mapped = pl.map(paramsA)
    result = sl.compute_series(mapped, C)
    assert result['a'] == C(7)
    assert result['b'] == C(17)
    assert result['c'] == C(27)
    assert result.index.name == 'dim_0'
    assert result.name == C


def test_compute_series_from_mapped_with_implicit_index() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    mapped = pl.map({A: [A(10 * i) for i in range(3)]})
    result = sl.compute_series(mapped, C)
    assert result[0] == C(7)
    assert result[1] == C(17)
    assert result[2] == C(27)
    assert result.index.name == 'dim_0'
    assert result.name == C


def test_compute_series_multiple_indices_creates_multiindex() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    paramsA = pd.DataFrame(
        {A: [A(10 * i) for i in range(3)]}, index=['a', 'b', 'c']
    ).rename_axis('x')
    paramsB = pd.DataFrame(
        {B: [B(i) for i in range(2)]}, index=['aa', 'bb']
    ).rename_axis('y')
    mapped = pl.map(paramsA).map(paramsB)
    result = sl.compute_series(mapped, C)
    assert result['a', 'aa'] == C(0)
    assert result['a', 'bb'] == C(1)
    assert result['b', 'aa'] == C(10)
    assert result['b', 'bb'] == C(11)
    assert result['c', 'aa'] == C(20)
    assert result['c', 'bb'] == C(21)
    assert result.index.names == list(mapped._cbgraph.index_names)
    assert result.name == C


def test_compute_series_ignores_unrelated_index() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    paramsA = pd.DataFrame(
        {A: [A(10 * i) for i in range(3)]}, index=['a', 'b', 'c']
    ).rename_axis('x')
    paramsB = pd.DataFrame(
        {B: [B(i) for i in range(2)]}, index=['aa', 'bb']
    ).rename_axis('y')
    mapped = pl.map(paramsA).map(paramsB)
    result = sl.compute_series(mapped, A)
    assert result.index.name == 'x'
    assert result['a'] == A(0)
    assert result['b'] == A(10)
    assert result['c'] == A(20)
    assert result.name == A


def test_compute_series_raises_if_node_is_not_mapped() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    mapped = pl.map({A: [A(10 * i) for i in range(3)]})
    with pytest.raises(ValueError, match='does not depend on any mapped nodes'):
        sl.compute_series(mapped, B)


def test_can_compute_subset_of_get_mapped_node_names() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    paramsA = pd.DataFrame(
        {A: [A(10 * i) for i in range(3)]}, index=['a', 'b', 'c']
    ).rename_axis('x')
    mapped = pl.map(paramsA)
    result = sl.get_mapped_node_names(mapped, C)
    # We lose the convenience of compute_series which returns a nicely setup series
    # but this is perfectly fine and possible.
    assert mapped.compute(result[1]) == A(17)
    assert mapped.compute(result.loc['b']) == A(17)
