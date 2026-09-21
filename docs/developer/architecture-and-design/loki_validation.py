"""Validate the Aggregation prototype on the esssans LoKI multi-run reduction.

Reference: with_pixel_mask_filenames + with_sample_runs + with_background_runs
(sciline map/reduce). Prototype: SansReduction, the object esssans would return
instead of a map/reduced pipeline, holding one Aggregation per run type, their
shared finalize stage, and the contributions; the pixel masks are a
list parameter read by one provider instead of an aggregation, since the point at
which they would be combined sits inside the per-run work.

Run from this directory with
    python loki_validation.py
"""

# ruff: noqa: T201

from __future__ import annotations

import time
from typing import Any, NewType

import ess.loki.data  # noqa: F401
import scipp as sc
from ess import loki, sans
from ess.sans.io import read_xml_detector_masking
from ess.sans.masking import apply_pixel_masks, to_detector_mask
from ess.sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    BeamCenter,
    CorrectedDetector,
    CorrectForGravity,
    Denominator,
    DetectorIDs,
    DetectorMasks,
    DirectBeam,
    EmptyBeamRun,
    Filename,
    MaskedDetectorIDs,
    NeXusDetectorName,
    NormalizedQ,
    NormalizedQxQy,
    Numerator,
    PixelMaskFilename,
    QBins,
    QxBins,
    QyBins,
    ReturnEvents,
    RunType,
    SampleRun,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBins,
    WavelengthDetector,
)
from ess.sans.workflow import _merge, merge_contributions
from scipp.testing import assert_allclose, assert_identical

import sciline
from sciline import Aggregation, Buffered, Stage, compute_members, warm

OUTPUTS = (BackgroundSubtractedIofQ, BackgroundSubtractedIofQxy)
SCHEDULER = sciline.scheduler.NaiveScheduler()

calls: dict[str, int] = {}


def _hit(name: str) -> None:
    calls[name] = calls.get(name, 0) + 1


# Counted copies of providers, with the original signatures so sciline sees them.
def counted_read_xml_detector_masking(filename: PixelMaskFilename) -> MaskedDetectorIDs:
    _hit('read_mask_file')
    return read_xml_detector_masking(filename)


def counted_to_detector_mask(
    ids: DetectorIDs, path: PixelMaskFilename, masked_ids: MaskedDetectorIDs
) -> DetectorMasks:
    _hit('to_detector_mask')
    return to_detector_mask(ids, path, masked_ids)


# The prototype's mask handling: one list parameter, the files read once (static),
# the masks built per run from the run's detector IDs.
PixelMaskFilenames = NewType('PixelMaskFilenames', tuple[str, ...])
MaskedDetectorIDsPerFile = NewType('MaskedDetectorIDsPerFile', dict[str, sc.Variable])


def read_mask_files(filenames: PixelMaskFilenames) -> MaskedDetectorIDsPerFile:
    return MaskedDetectorIDsPerFile(
        {f: counted_read_xml_detector_masking(PixelMaskFilename(f)) for f in filenames}
    )


def detector_masks(ids: DetectorIDs, masked: MaskedDetectorIDsPerFile) -> DetectorMasks:
    return _merge(
        *(
            counted_to_detector_mask(ids, PixelMaskFilename(p), m)
            for p, m in masked.items()
        )
    )


def counted_apply_pixel_masks(
    data: WavelengthDetector[RunType], masks: DetectorMasks
) -> CorrectedDetector[RunType, Numerator]:
    _hit('apply_pixel_masks')
    return apply_pixel_masks(data, masks)


def make_workflow() -> sciline.Pipeline:
    # Same as the `larmor_workflow` fixture in tests/loki/conftest.py, no_masks=False.
    wf: sciline.Pipeline = loki.LokiAtLarmorWorkflow()
    wf[NeXusDetectorName] = 'larmor_detector'
    wf[Filename[SampleRun]] = loki.data.loki_tutorial_sample_run_60339()
    wf[Filename[BackgroundRun]] = loki.data.loki_tutorial_background_run_60393()
    wf[Filename[TransmissionRun[SampleRun]]] = (
        loki.data.loki_tutorial_sample_transmission_run()
    )
    wf[Filename[TransmissionRun[BackgroundRun]]] = loki.data.loki_tutorial_run_60392()
    wf[Filename[EmptyBeamRun]] = loki.data.loki_tutorial_run_60392()
    wf[WavelengthBins] = sc.linspace(
        'wavelength', start=1.0, stop=13.0, num=51, unit='angstrom'
    )
    wf[CorrectForGravity] = True
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    wf[ReturnEvents] = False
    wf[QxBins] = sc.linspace('Qx', start=-0.3, stop=0.3, num=91, unit='1/angstrom')
    wf[QyBins] = sc.linspace('Qy', start=-0.2, stop=0.3, num=78, unit='1/angstrom')
    wf[QBins] = sc.linspace('Q', start=0.01, stop=0.3, num=101, unit='1/angstrom')
    wf[DirectBeam] = None
    for provider in (
        counted_read_xml_detector_masking,
        counted_to_detector_mask,
        counted_apply_pixel_masks,
    ):
        wf.insert(provider)
    return wf


def compare(name: str, a: Any, b: Any) -> None:
    try:
        assert_identical(a, b)
        print(f'  {name}: identical')
    except AssertionError:
        assert_allclose(a, b, rtol=sc.scalar(1e-12))
        print(f'  {name}: allclose (rtol 1e-12) but not identical')


def accumulators_for(run_type: type) -> dict[Any, Any]:
    return {
        qtype[run_type, part]: Buffered(merge_contributions)
        for part in (Numerator, Denominator)
        for qtype in (NormalizedQ, NormalizedQxQy)
    }


def show(label: str, t0: float) -> dict[str, int]:
    print(f'  {label}: {time.perf_counter() - t0:.1f} s, calls {calls}')
    return dict(calls)


class SansReduction:
    """What esssans would return instead of a map/reduced pipeline.

    Built from a pipeline with all parameters set; to change a parameter, build a
    new object. Holds one aggregation per run type, the finalize stage over the
    accumulation keys of both, and the contributions by filename.
    """

    run_types = (SampleRun, BackgroundRun)

    def __init__(self, pipeline: sciline.Pipeline) -> None:
        self._runs: dict[type, list[str]] = {rt: [] for rt in self.run_types}
        self._contributions: dict[type, dict[str, Any]] = {
            rt: {} for rt in self.run_types
        }
        self._aggregations = {
            rt: Aggregation(
                pipeline,
                members=(Filename[rt],),
                accumulators=accumulators_for(rt),
                scheduler=SCHEDULER,
            )
            for rt in self.run_types
        }
        self._finalize = Stage(
            pipeline,
            outputs=OUTPUTS,
            inputs=tuple(
                k for agg in self._aggregations.values() for k in agg.accumulation_keys
            ),
            scheduler=SCHEDULER,
        )

    def set_runs(self, run_type: type, runs: list[str]) -> None:
        self._runs[run_type] = list(runs)
        held = self._contributions[run_type]
        self._contributions[run_type] = {f: c for f, c in held.items() if f in runs}

    def compute(self) -> dict[Any, Any]:
        warm(
            *(agg.contribute_stage for agg in self._aggregations.values()),
            self._finalize,
        )
        combined: dict[Any, Any] = {}
        for run_type, agg in self._aggregations.items():
            held = self._contributions[run_type]
            for run in self._runs[run_type]:
                if run not in held:
                    held[run] = agg.contribute({Filename[run_type]: run})
            combined |= agg.combine(held.values())
        return self._finalize.compute(combined)


def main() -> None:
    masks = loki.data.loki_tutorial_mask_filenames()
    print(f'{len(masks)} mask file(s)')
    sample_runs = [
        loki.data.loki_tutorial_sample_run_60250(),
        loki.data.loki_tutorial_sample_run_60339(),
    ]
    background_runs = [
        loki.data.loki_tutorial_background_run_60248(),
        loki.data.loki_tutorial_background_run_60393(),
    ]

    base = make_workflow()
    # As in test_pipeline_can_compute_IofQ: beam center from the single-run workflow
    # with masks, then fixed as a parameter for both approaches.
    base[BeamCenter] = sans.beam_center_from_center_of_mass(
        sans.with_pixel_mask_filenames(base, masks)
    )
    calls.clear()

    # --- Reference: map/reduce ------------------------------------------------
    t0 = time.perf_counter()
    ref = sans.with_pixel_mask_filenames(base, masks)
    ref = sans.with_sample_runs(ref, runs=sample_runs)
    ref = sans.with_background_runs(ref, runs=background_runs)
    # Both sides run on the naive scheduler so that timings compare call structure
    # only. The default dask scheduler runs the members of one map/reduced graph in
    # threads, about 1.7 s faster here; over stages that parallelism is the caller's.
    ref_results = ref.compute(OUTPUTS, scheduler=SCHEDULER)
    t_ref = time.perf_counter() - t0
    ref_calls = show('reference', t0)
    calls.clear()

    # --- Prototype: the package object over two aggregations --------------------
    flat = base.copy()
    flat.insert(read_mask_files)
    flat.insert(detector_masks)
    flat[PixelMaskFilenames] = tuple(masks)
    # Filename[SampleRun] stays set on the pipeline: it is the member key of the
    # sample aggregation, and an ordinary parameter for the background aggregation,
    # whose DetectorMasks read the detector IDs of that one run (as in the reference).
    t0 = time.perf_counter()
    reduction = SansReduction(flat)
    reduction.set_runs(SampleRun, sample_runs[:1])
    reduction.set_runs(BackgroundRun, background_runs)
    reduction.compute()
    show('one sample run', t0)
    reduction.set_runs(SampleRun, sample_runs)
    results = reduction.compute()
    t_proto = time.perf_counter() - t0
    proto_calls = show('second sample run added', t0)

    print('final results, prototype vs reference:')
    for key in OUTPUTS:
        compare(key.__name__, results[key], ref_results[key])

    # --- Parameter change: a new object -----------------------------------------
    calls.clear()
    t0 = time.perf_counter()
    changed = flat.copy()
    changed[QBins] = sc.linspace('Q', start=0.01, stop=0.3, num=51, unit='1/angstrom')
    reduction = SansReduction(changed)
    reduction.set_runs(SampleRun, sample_runs)
    reduction.set_runs(BackgroundRun, background_runs)
    reduction.compute()
    show('QBins changed', t0)

    # --- Per-member intermediate ----------------------------------------------
    key = NormalizedQ[SampleRun, Numerator]
    members = compute_members(
        flat,
        members=(Filename[SampleRun],),
        key=key,
        table={i: {Filename[SampleRun]: f} for i, f in enumerate(sample_runs)},
    )
    single = sans.with_pixel_mask_filenames(base, masks)
    print('per-member NormalizedQ[SampleRun, Numerator] vs single-run compute:')
    for (m, value), filename in zip(members.items(), sample_runs, strict=True):
        single[Filename[SampleRun]] = filename
        compare(f'member {m}', value, single.compute(key))
    mapped = sciline.compute_mapped(ref, key)
    print('per-member vs sciline.compute_mapped:')
    for (m, value), (_, expected) in zip(members.items(), mapped.items(), strict=True):
        compare(f'member {m}', value, expected)

    # --- Three-entry-point form, as a framework would call it -------------------
    calls.clear()
    aggs = {
        rt: Aggregation(
            flat,
            members=(Filename[rt],),
            accumulators=accumulators_for(rt),
            scheduler=SCHEDULER,
        )
        for rt in (SampleRun, BackgroundRun)
    }
    finalize = Stage(
        flat,
        outputs=OUTPUTS,
        inputs=aggs[SampleRun].accumulation_keys
        + aggs[BackgroundRun].accumulation_keys,
        scheduler=SCHEDULER,
    )
    warm(*(agg.contribute_stage for agg in aggs.values()), finalize)
    sample = aggs[SampleRun].contribute({Filename[SampleRun]: sample_runs[0]})
    for run in sample_runs[1:]:
        sample = aggs[SampleRun].combine(
            [sample, aggs[SampleRun].contribute({Filename[SampleRun]: run})]
        )
    background = aggs[BackgroundRun].combine(
        aggs[BackgroundRun].contribute({Filename[BackgroundRun]: run})
        for run in background_runs
    )
    staged = finalize.compute({**sample, **background})
    print(f'three-entry-point form (calls {calls}):')
    for key in OUTPUTS:
        compare(key.__name__, staged[key], results[key])

    print(f'\nSUMMARY: reference {t_ref:.1f} s, prototype {t_proto:.1f} s')
    print(f'  reference calls: {ref_calls}')
    print(f'  prototype calls: {proto_calls}')


if __name__ == '__main__':
    main()
