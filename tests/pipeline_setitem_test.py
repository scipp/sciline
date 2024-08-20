# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import pytest

import sciline as sl

A = NewType('A', int)
B = NewType('B', int)
C = NewType('C', int)
D = NewType('D', int)
X = NewType('X', int)
Y = NewType('Y', int)

P = TypeVar('P', X, Y)


def a_to_b(a: A) -> B:
    return B(a + 1)


def ab_to_c(a: A, b: B) -> C:
    return C(a + b)


def b_to_c(b: B) -> C:
    return C(b + 2)


def c_to_d(c: C) -> D:
    return D(c + 4)


def test_setitem_can_compose_pipelines() -> None:
    ab = sl.Pipeline((a_to_b,))
    ab[A] = 0
    bc = sl.Pipeline((b_to_c,))
    bc[B] = ab
    assert bc.compute(C) == C(3)


def test_setitem_with_common_function_provider() -> None:
    def a() -> A:
        return A(2)

    abc = sl.Pipeline((a, ab_to_c))
    abc[B] = sl.Pipeline((a, a_to_b))
    assert abc.compute(C) == C(5)


def test_setitem_with_common_parameter_provider() -> None:
    abc = sl.Pipeline((ab_to_c,), params={A: A(3)})
    abc[B] = sl.Pipeline((a_to_b,), params={A: A(3)})
    assert abc.compute(C) == C(7)


def test_setitem_with_generic_providers() -> None:
    class GA(sl.Scope[P, int], int): ...

    class GB(sl.Scope[P, int], int): ...

    def ga_to_gb(ga: GA[P]) -> GB[P]:
        return GB[P](ga + 1)

    def gb_to_c(gbx: GB[X], gby: GB[Y]) -> C:
        return C(gbx + gby)

    abc = sl.Pipeline((gb_to_c,), params={GA[X]: 3, GA[Y]: 4})
    abc[GB[X]] = sl.Pipeline((ga_to_gb,), params={GA[X]: 3})[GB[X]]
    abc[GB[Y]] = sl.Pipeline((ga_to_gb,), params={GA[Y]: 4})[GB[Y]]
    assert abc.compute(C) == C(9)


def test_setitem_raises_if_value_pipeline_has_no_unique_output() -> None:
    abx = sl.Pipeline((a_to_b,))
    abx[X] = 666
    bc = sl.Pipeline((b_to_c,))
    with pytest.raises(ValueError, match='Graph must have exactly one sink node'):
        bc[B] = abx


def test_setitem_can_add_value_pipeline_at_new_node_given_by_key() -> None:
    ab = sl.Pipeline((a_to_b,))
    ab[A] = 0
    empty = sl.Pipeline(())
    empty[B] = ab
    assert empty.compute(B) == B(1)


def test_setitem_replaces_existing_node_value_with_value_pipeline() -> None:
    ab = sl.Pipeline((a_to_b,))
    ab[A] = 0
    bc = sl.Pipeline((b_to_c,))
    bc[B] = 111
    bc[B] = ab
    assert bc.compute(C) == C(3)


def test_setitem_replaces_existing_branch_with_value() -> None:
    ab = sl.Pipeline((a_to_b,))
    ab[A] = 0
    bc = sl.Pipeline((b_to_c,))
    bc[B] = ab
    bc[B] = 111
    assert bc.compute(C) == C(113)
    # __setitem__ prunes the entire branch, instead of just cutting the edges
    with pytest.raises(sl.UnsatisfiedRequirement):
        bc.compute(A)


def test_setitem_with_conflicting_nodes_in_value_pipeline_raises_on_data_mismatch() -> (
    None
):
    ab = sl.Pipeline((a_to_b,))
    ab[A] = 100
    abc = sl.Pipeline((ab_to_c,))
    abc[A] = 666
    with pytest.raises(ValueError, match="Node data differs"):
        abc[B] = ab


def test_setitem_with_conflicting_node_types_pipeline_raises_on_data_mismatch() -> None:
    def a() -> A:
        return A(100)

    ab = sl.Pipeline((a, a_to_b))
    abc = sl.Pipeline((ab_to_c,))
    abc[A] = 666
    with pytest.raises(ValueError, match="Node data differs"):
        abc[B] = ab


def test_setitem_with_conflicting_nodes_in_value_pipeline_accepts_on_data_match() -> (
    None
):
    ab = sl.Pipeline((a_to_b,))
    ab[A] = 100
    abc = sl.Pipeline((ab_to_c,))
    abc[A] = 100
    abc[B] = ab
    assert abc.compute(C) == C(201)


def test_setitem_with_conflicting_nodes_in_value_pipeline_raises_on_unique_data() -> (
    None
):
    ab = sl.Pipeline((a_to_b,))
    ab[A] = 100
    abc = sl.Pipeline((ab_to_c,))
    # Missing data is just as bad as conflicting data. At first glance, it might seem
    # like this should be allowed, but it would be a source of bugs and confusion.
    # In particular since it would allow for adding providers or parents to an
    # indirect dependency of the key in setitem, which would be very confusing.
    with pytest.raises(ValueError, match="Node data differs"):
        abc[B] = ab


def test_setitem_with_conflicting_node_inputs_in_value_pipeline_raises() -> None:
    def x_to_b(x: X) -> B:
        return B(x + 1)

    def bc_to_d(b: B, c: C) -> D:
        return D(b + c)

    xbc = sl.Pipeline((x_to_b, b_to_c))
    xbc[X] = 666
    abcd = sl.Pipeline((a_to_b, bc_to_d))
    abcd[A] = 100
    with pytest.raises(ValueError, match="Node inputs differ"):
        # If this worked naively by composing the NetworkX graph, it would look as
        # below, giving B 2 inputs instead of 1, even though provider has 1 argument,
        # corrupting the graph:
        # A  X
        # \ /
        #  B
        #  |\
        #  C |
        #  |/
        #  D
        abcd[C] = xbc
