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


def test_compute_mapped_single_index() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    paramsA = pd.DataFrame(
        {A: [A(10 * i) for i in range(3)]}, index=['a', 'b', 'c']
    ).rename_axis('x')
    mapped = pl.map(paramsA)
    result = sl.compute_mapped(mapped, C)
    assert result['a'] == C(7)
    assert result['b'] == C(17)
    assert result['c'] == C(27)
    assert result.index.name == 'x'
    assert result.name == C


def test_compute_mapped_single_index_with_no_name() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    paramsA = pd.DataFrame({A: [A(10 * i) for i in range(3)]}, index=['a', 'b', 'c'])
    mapped = pl.map(paramsA)
    result = sl.compute_mapped(mapped, C)
    assert result['a'] == C(7)
    assert result['b'] == C(17)
    assert result['c'] == C(27)
    assert result.index.name == 'dim_0'
    assert result.name == C


def test_compute_mapped_from_mapped_with_implicit_index() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    mapped = pl.map({A: [A(10 * i) for i in range(3)]})
    result = sl.compute_mapped(mapped, C)
    assert result[0] == C(7)
    assert result[1] == C(17)
    assert result[2] == C(27)
    assert result.index.name == 'dim_0'
    assert result.name == C


def test_compute_mapped_multiple_indices_creates_multiindex() -> None:
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
    result = sl.compute_mapped(mapped, C)
    assert result['a', 'aa'] == C(0)
    assert result['a', 'bb'] == C(1)
    assert result['b', 'aa'] == C(10)
    assert result['b', 'bb'] == C(11)
    assert result['c', 'aa'] == C(20)
    assert result['c', 'bb'] == C(21)
    assert result.index.names == list(mapped.index_names)
    assert result.name == C


def test_compute_mapped_ignores_unrelated_index() -> None:
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
    result = sl.compute_mapped(mapped, A)
    assert result.index.name == 'x'
    assert result['a'] == A(0)
    assert result['b'] == A(10)
    assert result['c'] == A(20)
    assert result.name == A


def test_compute_mapped_raises_if_node_is_not_mapped() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    mapped = pl.map({A: [A(10 * i) for i in range(3)]})
    with pytest.raises(ValueError, match='is not a mapped node'):
        sl.compute_mapped(mapped, B)


def test_compute_mapped_raises_if_node_depends_on_but_is_not_mapped() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    pl[B] = B(7)
    pl = pl.map({A: [A(10 * i) for i in range(3)]}).reduce(func=max, name=C)
    pl.insert(c_to_d)
    # Slightly different failure case from above: The relevant subgraph *does* have
    # indices, but the node does not.
    with pytest.raises(ValueError, match='is not a mapped node'):
        sl.compute_mapped(pl, D)


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
    # We lose the convenience of compute_mapped which returns a nicely setup series
    # but this is perfectly fine and possible.
    assert mapped.compute(result.iloc[1]) == A(17)
    assert mapped.compute(result.loc['b']) == A(17)


def test_compute_mapped_raises_if_multiple_mapped_nodes_with_given_name() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    pl = sl.Pipeline((ab_to_c,))
    paramsA = pd.DataFrame(
        {A: [A(10 * i) for i in range(3)]}, index=['a', 'b', 'c']
    ).rename_axis('x')
    paramsB = pd.DataFrame(
        {B: [B(i) for i in range(2)]}, index=['aa', 'bb']
    ).rename_axis('y')
    pl = pl.map(paramsA).map(paramsB).reduce(func=max, name=C, index='x')
    with pytest.raises(ValueError, match='Multiple mapped nodes with name'):
        sl.compute_mapped(pl, C)


def test_compute_mapped_with_partial_reduction_identifies_correct_index() -> None:
    def ab_to_c(a: A, b: B) -> C:
        return C(a + b)

    D = NewType('D', int)

    pl = sl.Pipeline((ab_to_c,))
    paramsA = pd.DataFrame(
        {A: [A(10 * i) for i in range(3)]}, index=['a', 'b', 'c']
    ).rename_axis('x')
    paramsB = pd.DataFrame(
        {B: [B(i) for i in range(2)]}, index=['aa', 'bb']
    ).rename_axis('y')
    pl = pl.map(paramsA).map(paramsB).reduce(func=max, name=D, index='x')
    result = sl.compute_mapped(pl, D)
    assert result['aa'] == C(20)
    assert result['bb'] == C(21)


def test_compute_mapped_index_names_selects_between_multiple_candidates() -> None:
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
    pl = mapped.reduce(func=max, name=C, index='x')
    result_y = sl.compute_mapped(pl, C, index_names=('y',))
    assert result_y['aa'] == C(20)
    assert result_y['bb'] == C(21)
    for index_names in [('x', 'y'), ('y', 'x')]:
        result_xy = sl.compute_mapped(pl, C, index_names=index_names)
        assert result_xy['a', 'aa'] == C(0)
        assert result_xy['a', 'bb'] == C(1)
        assert result_xy['b', 'aa'] == C(10)
        assert result_xy['b', 'bb'] == C(11)
        assert result_xy['c', 'aa'] == C(20)
        assert result_xy['c', 'bb'] == C(21)
