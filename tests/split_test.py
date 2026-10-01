# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from collections import Counter
from collections.abc import Hashable, Mapping
from typing import Any, NewType

import pytest

import sciline as sl
from sciline import Part, Reduced, Stage, split, warm
from sciline.reporter import Reporter
from sciline.typing import Graph, Key

A = NewType('A', int)  # outer loop
B = NewType('B', int)  # middle loop
C = NewType('C', int)  # inner loop
Offset = NewType('Offset', int)  # parameter
AValue = NewType('AValue', int)  # depends on A
BValue = NewType('BValue', int)  # depends on A and B
CValue = NewType('CValue', int)  # depends on A, B, and C; summed over C
Combined = NewType('Combined', int)  # per (A, B), from the sum over C


def a_value(a: A, offset: Offset) -> AValue:
    return AValue(10 * a + offset)


def b_value(x: AValue, b: B) -> BValue:
    return BValue(x + 100 * b)


def c_value(x: AValue, y: BValue, c: C) -> CValue:
    return CValue(x * y + c)


def combined(z: CValue, y: BValue, x: AValue) -> Combined:
    return Combined(z - y + x)


@pytest.fixture
def pipeline() -> sl.Pipeline:
    return sl.Pipeline([a_value, b_value, c_value, combined], params={Offset: 7})


@pytest.fixture
def parts() -> tuple[Part, Part, Part, Part]:
    a = Part(inputs=(A,))
    b = Part(inputs=(B,), parent=a)
    c = Part(inputs=(C,), outputs=(CValue,), parent=b)
    after_c = Part(inputs=(CValue,), outputs=(Combined,), parent=b)
    return a, b, c, after_c


def pick(stage: Stage, *contexts: Mapping[Key, Any]) -> dict[Key, Any]:
    values = {k: v for c in contexts for k, v in c.items()}
    return {k: values[k] for k in stage.inputs if k in values}


def test_value_is_computed_by_the_deepest_level_it_depends_on(
    pipeline: sl.Pipeline, parts: tuple[Part, ...]
) -> None:
    a, b, c, after_c = split(pipeline, *parts)
    assert a.outputs == (AValue,)
    assert set(b.inputs) == {AValue, B}
    assert b.outputs == (BValue,)
    # AValue skips the middle level: c reads it from a.
    assert set(c.inputs) == {AValue, BValue, C}
    assert c.outputs == (CValue,)
    assert set(after_c.inputs) == {AValue, BValue, CValue}


def test_nested_loops_give_the_result_of_flat_computes(
    pipeline: sl.Pipeline, parts: tuple[Part, ...]
) -> None:
    a, b, c, after_c = split(pipeline, *parts)
    warm(a, b, c, after_c)
    for a_ in (1, 2):
        a_out = a.compute({A: a_})
        for b_ in (3, 4):
            b_out = b.compute({**pick(b, a_out), B: b_})
            acc = Reduced[int](lambda x, y: x + y)()
            for c_ in (5, 6):
                acc.push(c.compute({**pick(c, a_out, b_out), C: c_})[CValue])
            result = after_c.compute({**pick(after_c, a_out, b_out), CValue: acc.value})

            p = pipeline.copy()
            p[A] = a_
            p[B] = b_
            total = 0
            for c_ in (5, 6):
                p[C] = c_
                total += p.compute(CValue)
            p[CValue] = total
            assert result[Combined] == p.compute(Combined)


def test_part_under_a_part_after_combining_reads_from_it(
    pipeline: sl.Pipeline, parts: tuple[Part, ...]
) -> None:
    # Combined depends on C through CValue. Cut at CValue, the input of after_c, it
    # does not, so a part under after_c reads it from there.
    D = NewType('D', int)
    DValue = NewType('DValue', int)

    def d_value(x: Combined, d: D) -> DValue:
        return DValue(x * d)

    pipeline.insert(d_value)
    d = Part(inputs=(D,), outputs=(DValue,), parent=parts[3])
    *_, after_c, d_stage = split(pipeline, *parts, d)
    assert Combined in after_c.outputs
    assert set(d_stage.inputs) == {Combined, D}


def test_value_that_depends_on_no_part_is_held_by_the_stage_that_reads_it(
    pipeline: sl.Pipeline, parts: tuple[Part, ...]
) -> None:
    a, *_ = split(pipeline, *parts)
    assert Offset in a.frontier
    assert Offset not in a.inputs


def test_output_that_does_not_vary_in_its_part_is_rejected(
    pipeline: sl.Pipeline,
) -> None:
    a = Part(inputs=(A,))
    b = Part(inputs=(B,), outputs=(BValue, AValue), parent=a)
    # Pushed per iteration of b, AValue would be counted once per value of B.
    with pytest.raises(ValueError, match=r'AValue.* on Part\(inputs=\(.*A,\)'):
        split(pipeline, a, b)


def test_output_that_depends_on_no_part_is_rejected(pipeline: sl.Pipeline) -> None:
    a = Part(inputs=(A,), outputs=(AValue, Offset))
    with pytest.raises(ValueError, match='Offset.* on none of its ancestors'):
        split(pipeline, a)


def test_reading_a_value_that_depends_on_a_non_ancestor_is_rejected(
    pipeline: sl.Pipeline,
) -> None:
    a = Part(inputs=(A,))
    b = Part(inputs=(B,), outputs=(BValue,), parent=a)
    orphan = Part(inputs=(C,), outputs=(CValue,))
    with pytest.raises(ValueError, match='not its ancestor'):
        split(pipeline, a, b, orphan)


def test_ancestors_must_be_among_the_parts(pipeline: sl.Pipeline) -> None:
    a = Part(inputs=(A,))
    b = Part(inputs=(B,), outputs=(BValue,), parent=a)
    with pytest.raises(ValueError, match='ancestor'):
        split(pipeline, b)


def test_part_without_outputs_that_no_part_reads_from_is_rejected(
    pipeline: sl.Pipeline,
) -> None:
    with pytest.raises(ValueError, match='no outputs'):
        split(pipeline, Part(inputs=(A,)))


def test_parts_compare_by_identity() -> None:
    assert Part(inputs=(A,)) != Part(inputs=(A,))


def test_split_uses_given_scheduler(
    pipeline: sl.Pipeline, parts: tuple[Part, ...]
) -> None:
    used: list[str] = []

    class Recording(sl.scheduler.NaiveScheduler):
        def get(
            self, graph: Graph, keys: list[Hashable], reporter: Reporter | None = None
        ) -> tuple[Any, ...]:
            used.append('get')
            return super().get(graph, keys, reporter)

    stages = split(pipeline, *parts, scheduler=Recording())
    for stage in stages:
        used.clear()
        stage.static()
        assert used


def test_per_iteration_work_runs_once_per_iteration_of_its_loop() -> None:
    calls: Counter[str] = Counter()

    def counted_a_value(a: A, offset: Offset) -> AValue:
        calls['a_value'] += 1
        return a_value(a, offset)

    def counted_b_value(x: AValue, b: B) -> BValue:
        calls['b_value'] += 1
        return b_value(x, b)

    pipeline = sl.Pipeline(
        [counted_a_value, counted_b_value, c_value], params={Offset: 7}
    )
    a = Part(inputs=(A,))
    b = Part(inputs=(B,), parent=a)
    c = Part(inputs=(C,), outputs=(CValue,), parent=b)
    a_stage, b_stage, c_stage = split(pipeline, a, b, c)
    for a_ in range(2):
        a_out = a_stage.compute({A: a_})
        for b_ in range(3):
            b_out = b_stage.compute({**pick(b_stage, a_out), B: b_})
            for c_ in range(4):
                c_stage.compute({**pick(c_stage, a_out, b_out), C: c_})
    assert calls == {'a_value': 2, 'b_value': 2 * 3}


def test_output_of_a_part_that_a_descendant_reads_is_output_once(
    pipeline: sl.Pipeline,
) -> None:
    a = Part(inputs=(A,))
    b = Part(inputs=(B,), outputs=(BValue,), parent=a)
    c = Part(inputs=(C,), outputs=(CValue,), parent=b)
    _, b_stage, c_stage = split(pipeline, a, b, c)
    assert b_stage.outputs == (BValue,)
    assert BValue in c_stage.inputs


def test_output_not_in_pipeline_is_rejected(pipeline: sl.Pipeline) -> None:
    Unknown = NewType('Unknown', int)
    with pytest.raises(ValueError, match='not in the pipeline'):
        split(pipeline, Part(inputs=(A,), outputs=(Unknown,)))


def test_input_not_needed_by_any_output_is_rejected(pipeline: sl.Pipeline) -> None:
    Unknown = NewType('Unknown', int)
    a = Part(inputs=(A,))
    b = Part(inputs=(B, Unknown), outputs=(BValue,), parent=a)
    with pytest.raises(ValueError, match='not needed'):
        split(pipeline, a, b)


def test_part_given_twice_is_rejected(
    pipeline: sl.Pipeline, parts: tuple[Part, ...]
) -> None:
    with pytest.raises(ValueError, match='once'):
        split(pipeline, *parts, parts[1])
