# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from typing import NewType

import pytest

import sciline as sl
from sciline import Stage, warm

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
    assert stage.compute({Numerator: [2.0, 4.0], Denominator: 3.0, Scale: 2.0})[IofQ] == 4.0
    assert calls.counts == {'normalize': 1}


def test_stage_rejects_input_the_outputs_do_not_need(pipeline: sl.Pipeline) -> None:
    with pytest.raises(ValueError, match='not needed'):
        Stage(pipeline, outputs=(Denominator,), inputs=(Scale,))


def test_stage_rejects_output_not_in_pipeline(pipeline: sl.Pipeline) -> None:
    with pytest.raises(ValueError, match='not in the pipeline'):
        Stage(pipeline, outputs=(Events,), inputs=(Filename,))


def test_stage_rejects_call_with_wrong_keys(pipeline: sl.Pipeline) -> None:
    stage = Stage(pipeline, outputs=(IofQ,), inputs=(Scale,))
    with pytest.raises(ValueError, match='Expected values for'):
        stage.compute({Scale: 1.0, Bins: 3})


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
    assert stage.compute({Numerator: [1.0]})[IofQ] == 1.0 / (sum(map(ord, 'unused')) * 4)


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


def test_stage_rejects_object_that_is_not_a_scheduler(pipeline: sl.Pipeline) -> None:
    with pytest.raises(ValueError, match='Scheduler'):
        Stage(
            pipeline,
            outputs=(Denominator,),
            inputs=(Filename,),
            scheduler=object(),  # type: ignore[arg-type]
        )


def test_warm_computes_shared_static_work_once(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    numerator = Stage(pipeline, outputs=(Numerator,), inputs=(Filename,))
    denominator = Stage(pipeline, outputs=(Denominator,), inputs=(Filename,))
    warm(numerator, denominator)
    assert numerator.static == {Calibration: 4.0, Bins: 2}
    assert denominator.static == {Calibration: 4.0}
    assert calls['calibration'] == 1


def test_warm_skips_stages_that_are_already_warm(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    numerator = Stage(pipeline, outputs=(Numerator,), inputs=(Filename,))
    denominator = Stage(pipeline, outputs=(Denominator,), inputs=(Filename,))
    numerator.static
    warm(numerator, denominator)
    warm(numerator, denominator)
    assert calls['calibration'] == 2


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
    assert finalize_stage.compute({Histogram: acc.value})[Result] == {1: 0.5, 2: 1.0, 3: 0.5}

    held.push(context_stage.compute({Angle: 3.0}))  # context update, accumulator kept
    acc.push(chunk_stage.compute({Events: [1.0], **held.value})[Histogram])
    assert finalize_stage.compute({Histogram: acc.value})[Result] == {1: 2.0, 2: 1.0, 3: 0.5}
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
