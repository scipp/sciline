"""StreamProcessor's shape expressed with Stage: chunk stage, held context, finalize."""

from __future__ import annotations

from typing import NewType

import sciline

from stage import Stage

Events = NewType('Events', list[float])  # dynamic: one chunk at a time
Angle = NewType('Angle', float)  # context: changes occasionally
Geometry = NewType('Geometry', list[float])  # depends on context only, cached
Histogram = NewType('Histogram', dict[int, float])  # accumulated
Norm = NewType('Norm', float)  # finalize parameter
Result = NewType('Result', dict[int, float])


class Calls:
    geometry = 0


def geometry(angle: Angle) -> Geometry:
    Calls.geometry += 1
    return Geometry([angle, 2 * angle])


def histogram(events: Events, geom: Geometry) -> Histogram:
    return Histogram({int(e): geom[0] for e in events})


def result(hist: Histogram, norm: Norm) -> Result:
    return Result({k: v / norm for k, v in hist.items()})


def add_hist(a: Histogram, b: Histogram) -> Histogram:
    return Histogram({k: a.get(k, 0) + b.get(k, 0) for k in a | b})


def test_stream_processor_shape():
    pipeline = sciline.Pipeline([geometry, histogram, result], params={Norm: 2.0})
    dynamic, context, accumulated, targets = (Events,), (Angle,), (Histogram,), (Result,)

    # Three stages, as StreamProcessor builds three sub-workflows today. The
    # context frontier, the context-dependent nodes just above the chunk-dependent
    # ones, is derived from the stages rather than declared.
    per_chunk = Stage(pipeline, outputs=accumulated, inputs=dynamic)
    context_frontier = Stage(
        pipeline, outputs=per_chunk.frontier, inputs=context
    ).dynamic_outputs
    assert context_frontier == (Geometry,)
    context_stage = Stage(pipeline, outputs=context_frontier, inputs=context)
    chunk_stage = Stage(pipeline, outputs=accumulated, inputs=dynamic + context_frontier)
    finalize_stage = Stage(pipeline, outputs=targets, inputs=accumulated)

    held = context_stage({Angle: 1.0})
    acc: Histogram | None = None
    for chunk in ([1.0, 2.0], [2.0, 3.0]):
        contribution = chunk_stage({Events: chunk, **held})[Histogram]
        acc = contribution if acc is None else add_hist(acc, contribution)
    assert Calls.geometry == 1
    assert finalize_stage({Histogram: acc})[Result] == {1: 0.5, 2: 1.0, 3: 0.5}

    held = context_stage({Angle: 3.0})  # context update, accumulator kept
    acc = add_hist(acc, chunk_stage({Events: [1.0], **held})[Histogram])
    assert finalize_stage({Histogram: acc})[Result] == {1: 2.0, 2: 1.0, 3: 0.5}
    assert Calls.geometry == 2
