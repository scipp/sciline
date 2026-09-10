"""Validate the Fold prototype on the esssans LoKI multi-run reduction.

Reference: with_pixel_mask_filenames + with_sample_runs + with_background_runs
(sciline map/reduce). Prototype: three sequential Folds on the flat pipeline.

Run with
    PYTHONPATH=<sciline-main>/src:/workspace/sciline/.scratch/proto python loki_validation.py
"""

from __future__ import annotations

import time
from typing import Any

import ess.loki.data  # noqa: F401
import sciline
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

import stage
from stage import Fold

OUTPUTS = (BackgroundSubtractedIofQ, BackgroundSubtractedIofQxy)

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


def counted_apply_pixel_masks(
    data: WavelengthDetector[RunType], masks: DetectorMasks
) -> CorrectedDetector[RunType, Numerator]:
    _hit('apply_pixel_masks')
    return apply_pixel_masks(data, masks)


def make_workflow() -> sciline.Pipeline:
    # Same as the `larmor_workflow` fixture in tests/loki/conftest.py, no_masks=False.
    wf = loki.LokiAtLarmorWorkflow()
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


def as_pipeline(fold: Fold) -> sciline.Pipeline:
    return fold.as_pipeline()


def cut_for(run_type: type) -> dict[Any, Any]:
    return {
        qtype[run_type, part]: merge_contributions
        for part in (Numerator, Denominator)
        for qtype in (NormalizedQ, NormalizedQxQy)
    }


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
    # Same scheduler as the prototype; the default dask scheduler runs the members
    # of one graph in threads and is about 1.7 s faster here.
    ref_results = ref.compute(OUTPUTS, scheduler=sciline.scheduler.NaiveScheduler())
    t_ref = time.perf_counter() - t0
    ref_calls = dict(calls)
    calls.clear()
    print(f'reference: {t_ref:.1f} s, calls {ref_calls}')

    # --- Prototype: three folds -----------------------------------------------
    t0 = time.perf_counter()
    masked = Fold(
        base,
        members={PixelMaskFilename: masks},
        at={DetectorMasks: _merge},
        outputs=OUTPUTS,
    )
    masked = as_pipeline(masked)
    t_masks = time.perf_counter() - t0
    print(f'  mask fold + as_pipeline: {t_masks:.1f} s, calls {calls}')
    sample_fold = Fold(
        masked,
        members={Filename[SampleRun]: sample_runs},
        at=cut_for(SampleRun),
        outputs=OUTPUTS,
    )
    with_samples = as_pipeline(sample_fold)
    t_samples = time.perf_counter() - t0 - t_masks
    print(f'  sample fold + as_pipeline: {t_samples:.1f} s, calls {calls}')
    bg_fold = Fold(
        with_samples,
        members={Filename[BackgroundRun]: background_runs},
        at=cut_for(BackgroundRun),
        outputs=OUTPUTS,
    )
    t_build = time.perf_counter() - t0
    print(f'  all folds built: {t_build:.1f} s, calls {calls}')
    results = bg_fold.compute()
    t_proto = time.perf_counter() - t0
    proto_calls = dict(calls)
    print(f'prototype: {t_proto:.1f} s total ({t_proto - t_build:.1f} s compute), calls {proto_calls}')

    print('final results, prototype vs reference:')
    for key in OUTPUTS:
        compare(key.__name__, results[key], ref_results[key])

    # --- Per-member intermediate ----------------------------------------------
    key = NormalizedQ[SampleRun, Numerator]
    members = sample_fold.compute_members(key)
    single = sans.with_pixel_mask_filenames(base, masks)
    print('per-member NormalizedQ[SampleRun, Numerator] vs single-run compute:')
    for (m, value), filename in zip(members.items(), sample_runs, strict=True):
        single[Filename[SampleRun]] = filename
        compare(f'member {m}', value, single.compute(key))
    try:
        mapped = sciline.compute_mapped(ref, key)
        print('per-member vs sciline.compute_mapped:')
        for (m, value), (_, expected) in zip(members.items(), mapped.items(), strict=True):
            compare(f'member {m}', value, expected)
    except Exception as e:  # noqa: BLE001
        print(f'  compute_mapped({key}) failed: {type(e).__name__}: {e}')

    # --- Three-stage form -----------------------------------------------------
    calls.clear()
    partial = bg_fold.contribute(0)
    for m in list(bg_fold.members)[1:]:
        partial = bg_fold.combine([partial, bg_fold.contribute(m)])
    staged = bg_fold.finalize(partial)
    print(f'three-stage form (calls {calls}):')
    for key in OUTPUTS:
        compare(key.__name__, staged[key], results[key])

    print(f'\nSUMMARY: reference {t_ref:.1f} s, prototype {t_proto:.1f} s')
    print(f'  reference calls: {ref_calls}')
    print(f'  prototype calls: {proto_calls}')


if __name__ == '__main__':
    main()
