# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Tests for generic providers using PEP 695 syntax (Python >= 3.12).

The type variables have no constraints; instantiations are inferred from the
concrete keys appearing in the pipeline. This file is excluded from collection
on Python < 3.12 via ``collect_ignore`` in ``conftest.py``.
"""

import pytest

import sciline as sl
from sciline.handler import UnsatisfiedRequirement

type A = int
type B = int


class Raw[Run](float): ...


class Processed[Run](float): ...


class Reduced[Run](float): ...


def process[Run](x: Raw[Run]) -> Processed[Run]:
    return Processed[Run](x * 2)


def reduce_run[Run](x: Processed[Run]) -> Reduced[Run]:
    return Reduced[Run](x + 1)


def test_generic_provider_instantiated_from_param() -> None:
    pl = sl.Pipeline([process], params={Raw[A]: Raw[A](1.5)})
    assert pl.compute(Processed[A]) == 3.0


def test_generic_provider_instantiated_per_requested_key() -> None:
    pl = sl.Pipeline([process], params={Raw[A]: Raw[A](1.0), Raw[B]: Raw[B](2.0)})
    assert pl.compute(Processed[A]) == 2.0
    assert pl.compute(Processed[B]) == 4.0


def test_chain_of_generic_providers() -> None:
    pl = sl.Pipeline([process, reduce_run], params={Raw[A]: Raw[A](2.0)})
    assert pl.compute(Reduced[A]) == 5.0


def test_generic_source_provider_instantiated_from_target() -> None:
    def make[Run]() -> Raw[Run]:
        return Raw[Run](7.0)

    pl = sl.Pipeline([make, process])
    assert pl.compute(Processed[A]) == 14.0


def test_concrete_provider_shadows_generic() -> None:
    def special() -> Processed[A]:
        return Processed[A](0.5)

    pl = sl.Pipeline([process, special], params={Raw[B]: Raw[B](1.0)})
    assert pl.compute(Processed[A]) == 0.5
    assert pl.compute(Processed[B]) == 2.0


def test_param_shadows_generic_provider() -> None:
    pl = sl.Pipeline(
        [process], params={Raw[A]: Raw[A](1.0), Processed[A]: Processed[A](5.0)}
    )
    assert pl.compute(Processed[A]) == 5.0


def test_later_generic_provider_with_equal_pattern_replaces_earlier() -> None:
    def process2[Run](x: Raw[Run]) -> Processed[Run]:
        return Processed[Run](x * 10)

    pl = sl.Pipeline([process, process2], params={Raw[A]: Raw[A](1.0)})
    assert pl.compute(Processed[A]) == 10.0


def test_more_specific_generic_provider_wins_regardless_of_order() -> None:
    def nested[Run](x: Raw[Run]) -> Processed[list[Run]]:
        return Processed[list[Run]](x * 100)

    for providers in ([process, nested], [nested, process]):
        pl = sl.Pipeline(providers, params={Raw[A]: Raw[A](1.0)})
        assert pl.compute(Processed[list[A]]) == 100.0
        assert pl.compute(Processed[A]) == 2.0


def test_constrained_generic_provider_is_more_specific_than_unconstrained() -> None:
    type C = int

    def special[Run: (A, B)](x: Raw[Run]) -> Processed[Run]:
        return Processed[Run](x * 10)

    for providers in ([process, special], [special, process]):
        pl = sl.Pipeline(providers, params={Raw[A]: Raw[A](1.0), Raw[C]: Raw[C](1.0)})
        assert pl.compute(Processed[A]) == 10.0
        assert pl.compute(Processed[C]) == 2.0


def test_incomparable_overlapping_generic_providers_raise() -> None:
    class Pair[T1, T2](float): ...

    def left[R]() -> Pair[A, R]:
        return Pair[A, R](1.0)

    def right[R]() -> Pair[R, B]:
        return Pair[R, B](2.0)

    pl = sl.Pipeline([left, right])
    assert pl.compute(Pair[A, A]) == 1.0
    assert pl.compute(Pair[B, B]) == 2.0
    with pytest.raises(sl.AmbiguousProvider, match='left|right'):
        pl.compute(Pair[A, B])


def test_multiple_typevars_bound_from_target() -> None:
    class Combined[R1, R2](float): ...

    def combine[R1, R2](x: Raw[R1], y: Processed[R2]) -> Combined[R1, R2]:
        return Combined[R1, R2](x + y)

    pl = sl.Pipeline(
        [combine, process], params={Raw[A]: Raw[A](1.0), Raw[B]: Raw[B](2.0)}
    )
    assert pl.compute(Combined[A, B]) == 5.0
    assert pl.compute(Combined[A, A]) == 3.0


def test_generic_type_alias() -> None:
    type RawImage[Run] = float
    type CleanImage[Run] = float

    def clean[Run](x: RawImage[Run]) -> CleanImage[Run]:
        return x + 1.0

    pl = sl.Pipeline([clean], params={RawImage[A]: 1.0})
    assert pl.compute(CleanImage[A]) == 2.0


def test_constrained_pep695_typevar_respects_declared_constraints() -> None:
    type C = int

    def process2[Run: (A, B)](x: Raw[Run]) -> Processed[Run]:
        return Processed[Run](x * 3)

    pl = sl.Pipeline(
        [process2],
        params={Raw[A]: Raw[A](1.0), Raw[B]: Raw[B](2.0), Raw[C]: Raw[C](3.0)},
    )
    assert pl.compute(Processed[A]) == 3.0
    assert pl.compute(Processed[B]) == 6.0
    with pytest.raises(UnsatisfiedRequirement):
        pl.compute(Processed[C])


def test_map_over_generic_pipeline() -> None:
    pl = sl.Pipeline([process])
    result = (
        pl.map({Raw[A]: [Raw[A](1.0), Raw[A](2.0)]})
        .reduce(func=lambda *v: sum(v), name='total')
        .compute('total')
    )
    assert result == 6.0


def test_getitem_returns_subgraph_with_instantiated_generics() -> None:
    pl = sl.Pipeline([process, reduce_run], params={Raw[A]: Raw[A](2.0)})
    sub = pl[Processed[A]]
    assert sub.compute(Processed[A]) == 4.0


def test_missing_dependency_of_instantiated_generic_raises() -> None:
    pl = sl.Pipeline([process])
    with pytest.raises(UnsatisfiedRequirement, match='Raw'):
        pl.compute(Processed[A])


def test_generic_param_applies_to_all_specializations() -> None:
    pl = sl.Pipeline([process], params={Raw: Raw(1.2)})
    assert pl.compute(Processed[A]) == pytest.approx(2.4)
    assert pl.compute(Processed[B]) == pytest.approx(2.4)


def test_concrete_param_shadows_generic_param() -> None:
    pl = sl.Pipeline([process], params={Raw: Raw(1.0), Raw[A]: Raw[A](5.0)})
    assert pl.compute(Processed[A]) == 10.0
    assert pl.compute(Processed[B]) == 2.0


def test_generic_param_used_by_mapped_pipeline() -> None:
    def combine(x: Processed[A], y: Raw[B]) -> float:
        return x + y

    pl = sl.Pipeline([process, combine], params={Raw: Raw(1.0)})
    result = (
        pl.map({Raw[A]: [Raw[A](1.0), Raw[A](2.0)]})
        .reduce(func=lambda *v: sum(v), name='total')
        .compute('total')
    )
    assert result == 8.0


def test_output_keys_include_derivable_generic_outputs() -> None:
    pl = sl.Pipeline([process, reduce_run], params={Raw[A]: Raw[A](1.0)})
    assert Reduced[A] in pl.output_keys()
    assert Reduced[B] not in pl.output_keys()
