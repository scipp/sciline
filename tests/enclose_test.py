# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from collections import Counter
from collections.abc import Hashable
from typing import Any, NewType

import pytest

import sciline as sl
from sciline import Reduced, Stage, enclose, warm
from sciline.reporter import Reporter
from sciline.typing import Graph

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
def stages(pipeline: sl.Pipeline) -> tuple[Stage, ...]:
    """Loops over A, then B, then C, with after_c inside the loop over B."""
    c = Stage(pipeline, outputs=(CValue,), inputs=(C,))
    after_c = Stage(pipeline, outputs=(Combined,), inputs=(CValue,))
    b, c, after_c = enclose(pipeline, [c, after_c], inputs=(B,))
    return enclose(pipeline, [b, c, after_c], inputs=(A,))


def test_value_is_computed_by_the_innermost_loop_whose_inputs_it_depends_on(
    stages: tuple[Stage, ...],
) -> None:
    a, b, c, after_c = stages
    assert a.outputs == (AValue,)
    assert set(b.inputs) == {AValue, B}
    assert b.outputs == (BValue,)
    # AValue skips the middle loop: c reads it from a.
    assert set(c.inputs) == {AValue, BValue, C}
    assert c.outputs == (CValue,)
    assert set(after_c.inputs) == {AValue, BValue, CValue}


def test_nested_loops_give_the_result_of_flat_computes(
    pipeline: sl.Pipeline, stages: tuple[Stage, ...]
) -> None:
    a, b, c, after_c = stages
    warm(*stages)
    for a_ in (1, 2):
        a_out = a.compute({A: a_})
        for b_ in (3, 4):
            b_out = b.compute({**a_out, B: b_})
            acc = Reduced[int](lambda x, y: x + y)()
            for c_ in (5, 6):
                acc.push(c.compute({**a_out, **b_out, C: c_})[CValue])
            result = after_c.compute({**a_out, **b_out, CValue: acc.value})

            p = pipeline.copy()
            p[A] = a_
            p[B] = b_
            total = 0
            for c_ in (5, 6):
                p[C] = c_
                total += p.compute(CValue)
            p[CValue] = total
            assert result[Combined] == p.compute(Combined)


def test_loop_inside_a_stage_after_combining_reads_from_it(
    pipeline: sl.Pipeline,
) -> None:
    D = NewType('D', int)
    DValue = NewType('DValue', int)

    def d_value(x: Combined, d: D) -> DValue:
        return DValue(x * d)

    pipeline.insert(d_value)
    d = Stage(pipeline, outputs=(DValue,), inputs=(D,))
    # Combined depends on C only through CValue, the accumulation key of the loop
    # over C, so a loop over D can run once per combined value.
    after_c, d = enclose(pipeline, [d], inputs=(CValue,))
    assert after_c.outputs == (Combined,)
    assert set(d.inputs) == {Combined, D}


def test_value_that_depends_on_no_loop_is_held_by_the_stage_that_reads_it(
    stages: tuple[Stage, ...],
) -> None:
    a, *_ = stages
    assert Offset in a.frontier
    assert Offset not in a.inputs


def test_output_that_does_not_vary_with_the_inputs_of_its_stage_is_rejected(
    pipeline: sl.Pipeline,
) -> None:
    b = Stage(pipeline, outputs=(BValue, AValue), inputs=(B,))
    # Pushed per iteration of the loop over B, AValue would be counted once per B.
    with pytest.raises(ValueError, match=r'AValue.*of stages\[0\]'):
        enclose(pipeline, [b], inputs=(A,))


def test_loop_that_no_stage_reads_from_is_rejected(pipeline: sl.Pipeline) -> None:
    after_c = Stage(pipeline, outputs=(Combined,), inputs=(CValue,))
    with pytest.raises(ValueError, match='No stage reads'):
        enclose(pipeline, [after_c], inputs=(C,))


def test_pipeline_changed_since_the_stages_were_built_is_rejected(
    pipeline: sl.Pipeline,
) -> None:
    c = Stage(pipeline, outputs=(CValue,), inputs=(C,))
    pipeline[Offset] = 8
    with pytest.raises(ValueError, match='differently'):
        enclose(pipeline, [c], inputs=(B,))


def test_outputs_of_the_loop_that_an_inner_stage_reads_are_output_once(
    pipeline: sl.Pipeline,
) -> None:
    c = Stage(pipeline, outputs=(CValue,), inputs=(C,))
    b, c = enclose(pipeline, [c], inputs=(B,), outputs=(BValue,))
    assert b.outputs == (BValue,)
    assert BValue in c.inputs


def test_output_not_in_pipeline_is_rejected(pipeline: sl.Pipeline) -> None:
    Unknown = NewType('Unknown', int)
    c = Stage(pipeline, outputs=(CValue,), inputs=(C,))
    with pytest.raises(ValueError, match='not in the pipeline'):
        enclose(pipeline, [c], inputs=(B,), outputs=(Unknown,))


def test_input_not_needed_by_any_output_is_rejected(pipeline: sl.Pipeline) -> None:
    Unknown = NewType('Unknown', int)
    c = Stage(pipeline, outputs=(CValue,), inputs=(C,))
    with pytest.raises(ValueError, match='not needed'):
        enclose(pipeline, [c], inputs=(B, Unknown))


def test_stage_left_out_of_a_loop_rejects_what_the_loop_computes(
    pipeline: sl.Pipeline,
) -> None:
    pipeline[A] = 1
    c = Stage(pipeline, outputs=(CValue,), inputs=(C,))
    b, c = enclose(pipeline, [c], inputs=(B,))
    # c is not passed, so it still holds AValue for the A set on the pipeline.
    a, b = enclose(pipeline, [b], inputs=(A,))
    a_out = a.compute({A: 2})
    b_out = b.compute({**a_out, B: 3})
    with pytest.raises(ValueError, match='does not take them as inputs'):
        c.compute({**a_out, **b_out, C: 4})


class Recording(sl.scheduler.NaiveScheduler):
    def __init__(self) -> None:
        self.calls = 0

    def get(
        self, graph: Graph, keys: list[Hashable], reporter: Reporter | None = None
    ) -> tuple[Any, ...]:
        self.calls += 1
        return super().get(graph, keys, reporter)


def test_outer_stage_uses_given_scheduler_and_inner_stages_keep_theirs(
    pipeline: sl.Pipeline,
) -> None:
    pipeline[A] = 1
    inner, outer = Recording(), Recording()
    c = Stage(pipeline, outputs=(CValue,), inputs=(C,), scheduler=inner)
    b, c = enclose(pipeline, [c], inputs=(B,), scheduler=outer)
    b_out = b.compute({B: 2})
    assert outer.calls > 0
    assert inner.calls == 0
    c.compute({**b_out, C: 3})
    assert inner.calls > 0


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
    c_stage = Stage(pipeline, outputs=(CValue,), inputs=(C,))
    b_stage, c_stage = enclose(pipeline, [c_stage], inputs=(B,))
    a_stage, b_stage, c_stage = enclose(pipeline, [b_stage, c_stage], inputs=(A,))
    for a_ in range(2):
        a_out = a_stage.compute({A: a_})
        for b_ in range(3):
            b_out = b_stage.compute({**a_out, B: b_})
            for c_ in range(4):
                c_stage.compute({**a_out, **b_out, C: c_})
    assert calls == {'a_value': 2, 'b_value': 2 * 3}


def test_visualize_stages_fills_what_each_stage_computes_per_call(
    stages: tuple[Stage, ...],
) -> None:
    from sciline.visualize import STAGE_FILLS

    lines: list[str] = sl.visualize_stages(*stages).source.splitlines()

    def style(key: type) -> str:
        # The last statement for the node is the one that sets its style.
        return next(
            line for line in reversed(lines) if line.lstrip().startswith(key.__name__)
        )

    for key, fill in zip(
        (AValue, BValue, CValue, Combined), STAGE_FILLS[:4], strict=True
    ):
        assert fill in style(key)
    assert any('Stage 1, per call with AValue, B' in line for line in lines)
