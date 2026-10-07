# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from collections.abc import Hashable
from typing import Any, NewType

import pytest

import sciline as sl
from sciline import Stage, StageSpec, build_stages
from sciline.reporter import Reporter
from sciline.typing import Graph
from sciline.visualize import DYNAMIC_STYLE, FRONTIER_STYLE, HELD_STYLE, INPUT_STYLE

Filename = NewType('Filename', str)
Mask = NewType('Mask', str)
Calibration = NewType('Calibration', float)  # static, expensive
Loaded = NewType('Loaded', list[float])  # per file
Masked = NewType('Masked', list[float])
Bins = NewType('Bins', int)
Numerator = NewType('Numerator', list[float])  # per file
Denominator = NewType('Denominator', float)  # per file
IofQ = NewType('IofQ', float)
Scale = NewType('Scale', float)  # cheap parameter, after Numerator and Denominator


class Calls:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def hit(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    def __getitem__(self, name: str) -> int:
        return self.counts.get(name, 0)

    def reset(self) -> None:
        self.counts.clear()


@pytest.fixture
def calls() -> Calls:
    return Calls()


@pytest.fixture
def pipeline(calls: Calls) -> sl.Pipeline:
    def calibration(mask: Mask) -> Calibration:
        calls.hit('calibration')
        return Calibration(float(len(mask)))

    def load(filename: Filename) -> Loaded:
        calls.hit('load')
        return Loaded([float(ord(c)) for c in filename])

    def apply_mask(data: Loaded, cal: Calibration) -> Masked:
        calls.hit('mask')
        return Masked([x * cal for x in data])

    def numerator(data: Masked, bins: Bins) -> Numerator:
        calls.hit('numerator')
        return Numerator(data[:bins])

    def denominator(data: Masked) -> Denominator:
        calls.hit('denominator')
        return Denominator(sum(data))

    def normalize(num: Numerator, den: Denominator, scale: Scale) -> IofQ:
        calls.hit('normalize')
        return IofQ(scale * sum(num) / den)

    return sl.Pipeline(
        [calibration, load, apply_mask, numerator, denominator, normalize],
        params={Mask: 'mask', Bins: 2, Scale: 1.0, Filename: 'unused'},
    )


def test_stage_computes_static_part_once_and_dynamic_part_per_call(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    stage = Stage(pipeline, outputs=(Numerator, Denominator), inputs=(Filename,))
    assert stage.frontier == (Calibration, Bins)
    assert set(stage.dynamic) == {Filename, Loaded, Masked, Numerator, Denominator}
    a = stage.compute({Filename: 'ab'})
    b = stage.compute({Filename: 'cde'})
    assert a[Numerator] == [97.0 * 4, 98.0 * 4]
    assert a[Denominator] == (97.0 + 98.0) * 4
    assert b[Denominator] == (99.0 + 100.0 + 101.0) * 4
    assert calls['calibration'] == 1
    assert calls['load'] == 2


def test_stage_with_intermediate_input_cuts_its_ancestors(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    stage = Stage(pipeline, outputs=(IofQ,), inputs=(Numerator, Denominator, Scale))
    assert stage.frontier == ()
    assert Filename not in stage.keys
    assert Calibration not in stage.keys
    assert (
        stage.compute({Numerator: [2.0, 4.0], Denominator: 3.0, Scale: 2.0})[IofQ]
        == 4.0
    )
    assert calls.counts == {'normalize': 1}


def test_stage_does_not_compute_ancestors_of_intermediate_input_shared_with_input(
    calls: Calls,
) -> None:
    def load(filename: Filename) -> Loaded:
        calls.hit('load')
        return Loaded([1.0])

    def apply_mask(data: Loaded) -> Masked:
        calls.hit('mask')
        return Masked(data)

    def denominator(data: Masked, filename: Filename) -> Denominator:
        return Denominator(sum(data) + len(filename))

    pipeline = sl.Pipeline([load, apply_mask, denominator])
    # The naive scheduler computes every node of the graph it is given.
    stage = Stage(
        pipeline,
        outputs=(Denominator,),
        inputs=(Masked, Filename),
        scheduler=sl.scheduler.NaiveScheduler(),
    )
    assert Loaded not in stage.keys
    assert stage.compute({Masked: [2.0], Filename: 'ab'})[Denominator] == 4.0
    assert calls.counts == {}


def test_stage_rejects_input_the_outputs_do_not_need(pipeline: sl.Pipeline) -> None:
    with pytest.raises(ValueError, match='not needed'):
        Stage(pipeline, outputs=(Denominator,), inputs=(Scale,))


def test_stage_rejects_input_needed_only_through_another_input(
    pipeline: sl.Pipeline,
) -> None:
    with pytest.raises(ValueError, match='not needed'):
        Stage(pipeline, outputs=(Denominator,), inputs=(Masked, Filename))


def test_stage_rejects_output_not_in_pipeline(pipeline: sl.Pipeline) -> None:
    with pytest.raises(ValueError, match='not in the pipeline'):
        Stage(pipeline, outputs=(Events,), inputs=(Filename,))


@pytest.mark.parametrize(
    'values',
    [{Numerator: [1.0]}, {Numerator: [1.0], Denominator: 1.0, Scale: 2.0}],
    ids=['missing', 'extra'],
)
def test_stage_rejects_call_with_wrong_keys(
    pipeline: sl.Pipeline, values: dict[type, Any]
) -> None:
    stage = Stage(pipeline, outputs=(IofQ,), inputs=(Numerator, Denominator))
    with pytest.raises(ValueError, match='Expected values for'):
        stage.compute(values)


def test_stage_holds_expensive_part_up_to_cheap_parameter(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    pipeline[Filename] = 'ab'
    expected = pipeline.compute(IofQ)
    calls.reset()
    stage = Stage(pipeline, outputs=(IofQ,), inputs=(Scale,))
    assert stage.frontier == (Numerator, Denominator)
    assert stage.compute({Scale: 1.0})[IofQ] == pytest.approx(expected)
    assert stage.compute({Scale: 3.0})[IofQ] == pytest.approx(3.0 * expected)
    assert calls['load'] == 1
    assert calls['normalize'] == 2


def test_stage_with_shared_ancestor_input_holds_the_rest(
    pipeline: sl.Pipeline,
) -> None:
    # Numerator is supplied; Denominator and Scale are static and held.
    stage = Stage(pipeline, outputs=(IofQ,), inputs=(Numerator,))
    assert stage.frontier == (Denominator, Scale)
    assert stage.compute({Numerator: [1.0]})[IofQ] == 1.0 / (
        sum(map(ord, 'unused')) * 4
    )


def test_stage_passes_through_an_output_that_is_an_input(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    stage = Stage(pipeline, outputs=(Numerator, IofQ), inputs=(Numerator,))
    assert stage.dynamic_outputs == (Numerator, IofQ)
    assert stage.compute({Numerator: [1.0]})[Numerator] == [1.0]
    assert calls['numerator'] == 0


def test_stage_is_a_snapshot_of_the_pipeline(pipeline: sl.Pipeline) -> None:
    stage = Stage(pipeline, outputs=(Numerator,), inputs=(Filename,))
    pipeline[Bins] = 1
    assert stage.compute({Filename: 'ab'})[Numerator] == [97.0 * 4, 98.0 * 4]


def test_stage_uses_given_scheduler(scheduler: sl.scheduler.Scheduler) -> None:
    # Builtin keys and local providers, so that the graph pickles by value for
    # dask.distributed workers, which cannot import this test module.
    def length(s: str) -> int:
        return len(s)

    def scaled(n: int, factor: float) -> complex:
        return complex(n * factor, 0.0)

    pipeline = sl.Pipeline([length, scaled], params={float: 0.5})
    stage = Stage(pipeline, outputs=(complex,), inputs=(str,), scheduler=scheduler)
    assert stage.frontier == (float,)
    assert stage.compute({str: 'abcd'})[complex] == 2.0 + 0.0j


def test_replacing_task_graph_dask_scheduler_changes_default_of_stage_and_pipeline(
    pipeline: sl.Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    used: list[str] = []

    class Recording(sl.scheduler.NaiveScheduler):
        def get(
            self, graph: Graph, keys: list[Hashable], reporter: Reporter | None = None
        ) -> tuple[Any, ...]:
            used.append('get')
            return super().get(graph, keys, reporter)

    # ESSlivedata selects its scheduler by replacing this name.
    monkeypatch.setattr(sl.task_graph, 'DaskScheduler', Recording)
    Stage(pipeline, outputs=(Denominator,), inputs=(Filename,)).compute(
        {Filename: 'ab'}
    )
    assert used
    used.clear()
    pipeline.compute(Calibration)
    assert used


def test_stage_rejects_object_that_is_not_a_scheduler(pipeline: sl.Pipeline) -> None:
    with pytest.raises(ValueError, match='Scheduler'):
        Stage(
            pipeline,
            outputs=(Denominator,),
            inputs=(Filename,),
            scheduler=object(),  # type: ignore[arg-type]
        )


def test_build_stages_computes_shared_static_work_once(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    numerator, denominator = build_stages(
        pipeline,
        [
            StageSpec(inputs=(Filename,), outputs=(Numerator,)),
            StageSpec(inputs=(Filename,), outputs=(Denominator,)),
        ],
    )
    assert calls['calibration'] == 1
    assert numerator.static() == {Calibration: 4.0, Bins: 2}
    assert denominator.static() == {Calibration: 4.0}
    numerator.compute({Filename: 'ab'})
    denominator.compute({Filename: 'ab'})
    assert calls['calibration'] == 1


def test_build_stages_returns_stages_in_order_of_specs(pipeline: sl.Pipeline) -> None:
    specs = [
        StageSpec(inputs=(Filename,), outputs=(Numerator, Denominator)),
        StageSpec(inputs=(Numerator, Denominator), outputs=(IofQ,)),
    ]
    stages = build_stages(pipeline, specs)
    assert [(s.inputs, s.outputs) for s in stages] == [
        (spec.inputs, spec.outputs) for spec in specs
    ]


def test_build_stages_with_missing_parameter_raises_unsatisfied() -> None:
    def calibration(mask: Mask) -> Calibration:
        return Calibration(float(len(mask)))

    def numerator(cal: Calibration, filename: Filename) -> Numerator:
        return Numerator([cal])

    pipeline = sl.Pipeline([calibration, numerator])
    with pytest.raises(sl.UnsatisfiedRequirement):
        build_stages(pipeline, [StageSpec(inputs=(Filename,), outputs=(Numerator,))])


def test_build_stages_rejects_stage_holding_a_value_from_a_parameter_another_varies(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    with pytest.raises(
        ValueError, match=r'specs\[1\] holds .*Filename.*which specs\[0\] takes'
    ):
        build_stages(
            pipeline,
            [
                StageSpec(inputs=(Filename,), outputs=(Denominator,)),
                # Holds Masked, loaded from the Filename set on the pipeline.
                StageSpec(inputs=(Bins,), outputs=(Numerator,)),
            ],
        )
    assert calls['load'] == 0


def test_build_stages_allows_stage_taking_a_value_that_another_stage_holds(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    build_stages(
        pipeline,
        [
            StageSpec(inputs=(Filename,), outputs=(Calibration, Loaded)),
            StageSpec(inputs=(Loaded, Calibration), outputs=(Masked,)),
        ],
    )
    assert calls['calibration'] == 1


def test_build_stages_rejects_invalid_spec(pipeline: sl.Pipeline) -> None:
    with pytest.raises(ValueError, match='not needed'):
        build_stages(pipeline, [StageSpec(inputs=(Scale,), outputs=(Numerator,))])


# A stream: chunks of events are histogrammed against a geometry that depends on
# a context value which changes occasionally; histograms are accumulated.

Events = NewType('Events', list[float])
Angle = NewType('Angle', float)
Geometry = NewType('Geometry', list[float])
Histogram = NewType('Histogram', dict[int, float])
Norm = NewType('Norm', float)
Result = NewType('Result', dict[int, float])


class Latest:
    """Holds the latest value pushed."""

    def __init__(self) -> None:
        self.value: dict[type, float] = {}

    def push(self, value: dict[type, float]) -> None:
        self.value = value


def add_histograms(left: Histogram, right: Histogram) -> Histogram:
    return Histogram({k: left.get(k, 0) + right.get(k, 0) for k in left | right})


def test_stream_of_chunks_with_context_held_between_changes(calls: Calls) -> None:
    def geometry(angle: Angle) -> Geometry:
        calls.hit('geometry')
        return Geometry([angle, 2 * angle])

    def histogram(events: Events, geom: Geometry) -> Histogram:
        return Histogram({int(e): geom[0] for e in events})

    def result(hist: Histogram, norm: Norm) -> Result:
        return Result({k: v / norm for k, v in hist.items()})

    pipeline = sl.Pipeline([geometry, histogram, result], params={Norm: 2.0})

    # The context frontier, the context-dependent keys just above the
    # chunk-dependent ones, is derived from the stages rather than declared.
    per_chunk = Stage(pipeline, outputs=(Histogram,), inputs=(Events,))
    context_frontier = Stage(
        pipeline, outputs=per_chunk.frontier, inputs=(Angle,)
    ).dynamic_outputs
    assert context_frontier == (Geometry,)
    context_stage = Stage(pipeline, outputs=context_frontier, inputs=(Angle,))
    chunk_stage = Stage(
        pipeline, outputs=(Histogram,), inputs=(Events, *context_frontier)
    )
    finalize_stage = Stage(pipeline, outputs=(Result,), inputs=(Histogram,))

    held = Latest()
    acc = sl.Reduced(add_histograms)()

    held.push(context_stage.compute({Angle: 1.0}))
    for chunk in ([1.0, 2.0], [2.0, 3.0]):
        acc.push(chunk_stage.compute({Events: chunk, **held.value})[Histogram])
    assert calls['geometry'] == 1
    assert finalize_stage.compute({Histogram: acc.value})[Result] == {
        1: 0.5,
        2: 1.0,
        3: 0.5,
    }

    held.push(context_stage.compute({Angle: 3.0}))  # context update, accumulator kept
    acc.push(chunk_stage.compute({Events: [1.0], **held.value})[Histogram])
    assert finalize_stage.compute({Histogram: acc.value})[Result] == {
        1: 2.0,
        2: 1.0,
        3: 0.5,
    }
    assert calls['geometry'] == 2


def test_stage_called_from_threads_computes_static_part_once(calls: Calls) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from time import sleep

    def calibration(mask: Mask) -> Calibration:
        calls.hit('calibration')
        sleep(0.05)
        return Calibration(float(len(mask)))

    def load(filename: Filename, cal: Calibration) -> Loaded:
        return Loaded([cal * ord(c) for c in filename])

    pipeline = sl.Pipeline([calibration, load], params={Mask: 'mask'})
    stage = Stage(pipeline, outputs=(Loaded,), inputs=(Filename,))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(stage.compute, [{Filename: 'ab'}] * 4))
    assert calls['calibration'] == 1
    assert all(r == results[0] for r in results)


def node_line(source: str, key: type, attr: str = '') -> str:
    """The last statement for the node of ``key`` that sets ``attr``.

    Without ``attr``, this is the statement that sets the node's style.
    """
    return next(
        line
        for line in reversed(source.splitlines())
        if line.lstrip('\t').startswith(f'{key.__name__} [{attr}')
    )


def test_stage_visualize_marks_inputs_held_and_per_call_nodes(
    pipeline: sl.Pipeline,
) -> None:
    stage = Stage(pipeline, outputs=(Numerator,), inputs=(Filename,))
    source: str = stage.visualize().source
    assert INPUT_STYLE['fillcolor'] in node_line(source, Filename)
    assert HELD_STYLE['fillcolor'] in node_line(source, Mask)
    assert FRONTIER_STYLE['penwidth'] in node_line(source, Calibration)
    assert DYNAMIC_STYLE['fillcolor'] in node_line(source, Masked)
    assert 'peripheries=2' in node_line(source, Numerator)
    assert 'cluster_legend' in source


def test_stage_visualize_without_legend(pipeline: sl.Pipeline) -> None:
    stage = Stage(pipeline, outputs=(Numerator,), inputs=(Filename,))
    assert 'cluster_legend' not in stage.visualize(show_legend=False).source


def test_stage_visualize_can_hide_held_ancestors(pipeline: sl.Pipeline) -> None:
    stage = Stage(pipeline, outputs=(Numerator,), inputs=(Filename,))
    source: str = stage.visualize(show_held_ancestors=False).source
    assert FRONTIER_STYLE['penwidth'] in node_line(source, Calibration)
    assert 'Mask [' not in source
    assert 'Computed once, not kept' not in source


def test_visualize_stages_styles_groups_given_by_caller(pipeline: sl.Pipeline) -> None:
    per_file = Stage(pipeline, outputs=(Numerator,), inputs=(Filename,))
    final = Stage(pipeline, outputs=(IofQ,), inputs=(Numerator,))
    source: str = sl.visualize_stages(
        per_file,
        final,
        groups={'Context': ({'fillcolor': '#123456'}, (Calibration,))},
    ).source
    assert '#123456' in node_line(source, Calibration)
    assert 'Context' in source
    # Numerator is computed by per_file, so it is drawn with its provider.
    assert 'via:' in node_line(source, Numerator, 'label=')


def test_visualize_stages_fills_what_each_stage_computes_per_call(
    pipeline: sl.Pipeline,
) -> None:
    from sciline.visualize import STAGE_FILLS

    per_file = Stage(pipeline, outputs=(Numerator, Denominator), inputs=(Filename,))
    final = Stage(pipeline, outputs=(IofQ,), inputs=(Numerator, Denominator))
    source: str = sl.visualize_stages(per_file, final).source
    assert STAGE_FILLS[0] in node_line(source, Masked)
    assert STAGE_FILLS[1] in node_line(source, IofQ)
    assert 'Stage 1, per call with Numerator, Denominator' in source
