"""StreamProcessor's shape: stages connected by a Forwarder and a reducer."""

from __future__ import annotations

from typing import NewType

from stage import Forwarder, Stage

import sciline

Events = NewType('Events', list[float])  # dynamic: one chunk at a time
Angle = NewType('Angle', float)  # context: changes occasionally
Geometry = NewType('Geometry', list[float])  # depends on context only, held
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


class Summed:
    """A reducer: ess.reduce's EternalAccumulator, for dicts."""

    def __init__(self) -> None:
        self.value: Histogram = Histogram({})

    def push(self, hist: Histogram) -> None:
        self.value = Histogram(
            {k: self.value.get(k, 0) + hist.get(k, 0) for k in self.value | hist}
        )


def test_stream_processor_shape():
    pipeline = sciline.Pipeline([geometry, histogram, result], params={Norm: 2.0})
    dynamic, context, accumulated, targets = (
        (Events,),
        (Angle,),
        (Histogram,),
        (Result,),
    )

    # Three stages, as StreamProcessor builds three sub-workflows today. The
    # context frontier, the context-dependent nodes just above the chunk-dependent
    # ones, is derived from the stages rather than declared.
    per_chunk = Stage(pipeline, outputs=accumulated, inputs=dynamic)
    context_frontier = Stage(
        pipeline, outputs=per_chunk.frontier, inputs=context
    ).dynamic_outputs
    assert context_frontier == (Geometry,)
    context_stage = Stage(pipeline, outputs=context_frontier, inputs=context)
    chunk_stage = Stage(
        pipeline, outputs=accumulated, inputs=dynamic + context_frontier
    )
    finalize_stage = Stage(pipeline, outputs=targets, inputs=accumulated)

    # The connectors say why each cut is there: the context is held between
    # changes, the histogram is accumulated.
    held = Forwarder()
    acc = Summed()

    held.push(context_stage({Angle: 1.0}))
    for chunk in ([1.0, 2.0], [2.0, 3.0]):
        acc.push(chunk_stage({Events: chunk, **held.value})[Histogram])
    assert Calls.geometry == 1
    assert finalize_stage({Histogram: acc.value})[Result] == {1: 0.5, 2: 1.0, 3: 0.5}

    held.push(context_stage({Angle: 3.0}))  # context update, accumulator kept
    acc.push(chunk_stage({Events: [1.0], **held.value})[Histogram])
    assert finalize_stage({Histogram: acc.value})[Result] == {1: 2.0, 2: 1.0, 3: 0.5}
    assert Calls.geometry == 2
