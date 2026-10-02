"""Validate enclose on the esssans LoKI reduction over runs times detector banks.

Reference: with_banks(with_sample_runs(...)) (sciline map/reduce), which gives
IntensityQ[SampleRun] per bank, combined over the sample runs. esssans does not
combine over banks, since banks have different Q resolution.
Prototype: a stage per bank from NeXusDetectorName to the accumulation keys, enclosed
in a loop over Filename[SampleRun], and a final stage from the accumulation keys to
IntensityQ[SampleRun]. The driver keeps one set of accumulators per bank.

The only multi-bank test file is loki_coda_file. The second sample run is a copy with
the detector event IDs reversed and the monitor event time offsets scaled, so that
both the per-bank and the per-run values differ between the runs. A value that
depends on the run but is held by the bank stage would then give a wrong result.

Run from this directory with
    python loki_banks_validation.py
"""

# ruff: noqa: T201, S101

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import ess.loki.data  # noqa: F401
import h5py
import numpy as np
import scipp as sc
from ess import loki, sans
from ess.reduce.nexus.types import NeXusData
from ess.reduce.nexus.workflow import assemble_detector_data
from ess.sans.normalization import norm_monitor_term
from ess.sans.types import (
    BeamCenter,
    CorrectedMonitor,
    CorrectForGravity,
    Denominator,
    DetectorMasks,
    DirectBeam,
    EmptyBeamRun,
    EmptyDetector,
    Filename,
    Incident,
    IntensityQ,
    LookupTableFilename,
    MonitorTerm,
    NeXusDetectorName,
    NormalizedQ,
    Numerator,
    QBins,
    RawDetector,
    ReturnEvents,
    SampleRun,
    TransmissionFraction,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBins,
)
from ess.sans.workflow import merge_contributions
from scipp.testing import assert_allclose, assert_identical
from scippnexus import NXdetector

import sciline
from sciline import Buffered, Stage, enclose, warm

TARGET = IntensityQ[SampleRun]
ACCUMULATED = (NormalizedQ[SampleRun, Numerator], NormalizedQ[SampleRun, Denominator])
BANKS = [f'loki_detector_{i}' for i in range(9)]
SCHEDULER = sciline.scheduler.NaiveScheduler()

calls: dict[str, int] = {}


def _hit(name: str) -> None:
    calls[name] = calls.get(name, 0) + 1


# Counted copies of providers, with the signatures for SampleRun only, so that the
# counts are per sample run (monitor term) and per sample run and bank (detector).
def counted_norm_monitor_term(
    incident_monitor: CorrectedMonitor[SampleRun, Incident],
    transmission_fraction: TransmissionFraction[SampleRun],
) -> MonitorTerm[SampleRun]:
    _hit('monitor_term')
    return norm_monitor_term(incident_monitor, transmission_fraction)


def counted_assemble_detector_data(
    detector: EmptyDetector[SampleRun], neutron_data: NeXusData[NXdetector, SampleRun]
) -> RawDetector[SampleRun]:
    _hit('assemble_detector')
    return assemble_detector_data(detector, neutron_data)


def make_workflow() -> sciline.Pipeline:
    # As in the esssans user guide loki-reduction-ess.ipynb, except for the wavelength
    # bins. The test file holds 5 pulses, and with the 200 bins of the guide some
    # bins have no monitor counts. The transmission fraction is NaN there, and since
    # I(Q) sums over wavelength, every value of I(Q) would be NaN.
    wf: sciline.Pipeline = loki.LokiWorkflow()
    wf[WavelengthBins] = sc.linspace('wavelength', 1.0, 10.0, 21, unit='angstrom')
    wf[QBins] = sc.linspace('Q', start=0.01, stop=0.3, num=101, unit='1/angstrom')
    wf[CorrectForGravity] = True
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    wf[ReturnEvents] = False
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[DirectBeam] = None
    wf[DetectorMasks] = {}
    wf[LookupTableFilename] = loki.data.loki_lookup_table_no_choppers()
    wf[Filename[EmptyBeamRun]] = loki.data.loki_coda_file()
    wf[Filename[TransmissionRun[SampleRun]]] = loki.data.loki_coda_file()
    wf.insert(counted_norm_monitor_term)
    wf.insert(counted_assemble_detector_data)
    return wf


def make_second_run(directory: str) -> str:
    """Copy the test file with different detector and monitor events."""
    path = str(Path(directory) / 'second_run.hdf')
    shutil.copy(loki.data.loki_coda_file(), path)
    with h5py.File(path, 'r+') as f:
        instrument = f['entry/instrument']
        for bank in BANKS:
            ids = instrument[f'{bank}/detector_events/event_id']
            ids[...] = ids[()][::-1]
        for name in instrument:
            if name.startswith('beam_monitor'):
                offset = instrument[f'{name}/monitor_events/event_time_offset']
                offset[...] = (offset[()] * 0.9).astype(offset.dtype)
    return path


def compare(name: str, a: sc.DataArray, b: sc.DataArray) -> None:
    # assert_identical treats NaN as equal, so check that there is something to compare.
    # NaN remains in Q bins without counts.
    finite = np.isfinite(a.values).mean()
    assert finite > 0, name
    try:
        assert_identical(a, b)
        print(f'  {name}: identical ({finite:.0%} finite)')
    except AssertionError:
        assert_allclose(a, b, rtol=sc.scalar(1e-12))
        print(
            f'  {name}: allclose (rtol 1e-12) but not identical ({finite:.0%} finite)'
        )


def show(label: str, t0: float) -> dict[str, int]:
    print(f'  {label}: {time.perf_counter() - t0:.1f} s, calls {calls}')
    return dict(calls)


def main() -> None:
    wf = make_workflow()
    with tempfile.TemporaryDirectory() as tmp:
        runs = [str(loki.data.loki_coda_file()), make_second_run(tmp)]

        # --- Reference: map/reduce ------------------------------------------------
        calls.clear()
        t0 = time.perf_counter()
        ref = sans.with_banks(sans.with_sample_runs(wf, runs=runs), banks=BANKS)
        # compute_mapped takes no scheduler, so compute the mapped nodes directly.
        names = sciline.get_mapped_node_names(ref, TARGET)
        computed = ref.compute(names, scheduler=SCHEDULER)
        ref_results = {bank: computed[name] for bank, name in names.items()}
        t_ref = time.perf_counter() - t0
        ref_calls = show('reference', t0)

        # --- Prototype: a bank stage enclosed in a loop over runs --------------------
        calls.clear()
        t0 = time.perf_counter()
        bank_stage = Stage(
            wf, outputs=ACCUMULATED, inputs=(NeXusDetectorName,), scheduler=SCHEDULER
        )
        run_stage, bank_stage = enclose(
            wf, [bank_stage], inputs=(Filename[SampleRun],), scheduler=SCHEDULER
        )
        final = Stage(wf, outputs=(TARGET,), inputs=ACCUMULATED, scheduler=SCHEDULER)
        warm(run_stage, bank_stage, final)
        acc = {
            b: {k: Buffered(merge_contributions)() for k in ACCUMULATED} for b in BANKS
        }
        held_per_run: dict[str, dict[Any, Any]] = {}
        per_run: dict[str, dict[str, Any]] = {}
        for run in runs:
            held = run_stage.compute({Filename[SampleRun]: run})
            held_per_run[run] = held
            for bank in BANKS:
                values = bank_stage.compute({**held, NeXusDetectorName: bank})
                per_run.setdefault(bank, {})[run] = values
                for key in bank_stage.dynamic_outputs:
                    acc[bank][key].push(values[key])
        results = {
            b: final.compute({k: a.value for k, a in acc[b].items()})[TARGET]
            for b in BANKS
        }
        t_proto = time.perf_counter() - t0
        proto_calls = show('prototype', t0)

        print('forwarded by enclose (run_stage.outputs):')
        for key in run_stage.outputs:
            print(f'  {key}')

        print('the two runs differ in the monitor term and in every bank:')
        a, b = (held[MonitorTerm[SampleRun]] for held in held_per_run.values())
        assert not sc.identical(a, b)
        numerator = ACCUMULATED[0]
        for bank, values in per_run.items():
            a, b = (v[numerator] for v in values.values())
            assert not sc.identical(a, b), bank
        print('  yes')

        print('per-bank results, prototype vs reference:')
        for bank in BANKS:
            compare(bank, results[bank], ref_results[bank])

    print(f'\nSUMMARY: reference {t_ref:.1f} s, prototype {t_proto:.1f} s')
    print(f'  reference calls: {ref_calls}')
    print(f'  prototype calls: {proto_calls}')


if __name__ == '__main__':
    main()
