from __future__ import annotations

from typing import NewType, TypeVar

import pandas as pd
import pytest
import sciline

from stage import Fold, Stage, compute_members, warm

# --- A small "reduction" graph, shaped like esssans -----------------------------

Filename = NewType('Filename', str)
Mask = NewType('Mask', str)
Calibration = NewType('Calibration', float)  # static, expensive
Loaded = NewType('Loaded', list[float])  # per file
Masked = NewType('Masked', list[float])
Bins = NewType('Bins', int)
Numerator = NewType('Numerator', list[float])  # per file, additive by concat
Denominator = NewType('Denominator', float)  # per file, additive by sum
IofQ = NewType('IofQ', float)
Scale = NewType('Scale', float)  # cheap parameter, after the cut


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
def pipeline(calls: Calls) -> sciline.Pipeline:
    def calibration(mask: Mask) -> Calibration:
        calls.hit('calibration')
        return float(len(mask))

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

    return sciline.Pipeline(
        [calibration, load, apply_mask, numerator, denominator, normalize],
        params={Mask: 'mask', Bins: 2, Scale: 1.0, Filename: 'unused'},
    )


def concat(*parts: list[float]) -> list[float]:
    return [x for part in parts for x in part]


def add(*parts: float) -> float:
    return sum(parts)


CUT = {Numerator: concat, Denominator: add}


def reference(pipeline: sciline.Pipeline, files: list[str], scale: float = 1.0):
    num, den = [], 0.0
    for f in files:
        p = pipeline.copy()
        p[Filename] = f
        num += p.compute(Numerator)
        den += p.compute(Denominator)
    return scale * sum(num) / den


def _with(pipeline, key, value):
    p = pipeline.copy()
    p[key] = value
    return p


# --- Stage ----------------------------------------------------------------------


def test_stage_computes_static_part_once_and_dynamic_part_per_call(pipeline, calls):
    stage = Stage(pipeline, outputs=(Numerator, Denominator), inputs=(Filename,))
    assert stage.frontier == (Calibration, Bins)
    assert set(stage.dynamic) == {Filename, Loaded, Masked, Numerator, Denominator}
    a = stage({Filename: 'ab'})
    b = stage({Filename: 'cde'})
    assert a[Numerator] == [97.0 * 4, 98.0 * 4]
    assert a[Denominator] == (97.0 + 98.0) * 4
    assert b[Denominator] == (99.0 + 100.0 + 101.0) * 4
    assert calls['calibration'] == 1
    assert calls['load'] == 2


def test_stage_with_intermediate_input_cuts_its_ancestors(pipeline, calls):
    stage = Stage(pipeline, outputs=(IofQ,), inputs=(Numerator, Denominator, Scale))
    assert stage.frontier == ()
    assert stage({Numerator: [2.0, 4.0], Denominator: 3.0, Scale: 2.0})[IofQ] == 4.0
    assert calls.counts == {'normalize': 1}


def test_stage_rejects_input_the_outputs_do_not_need(pipeline):
    with pytest.raises(ValueError, match='not needed'):
        Stage(pipeline, outputs=(Denominator,), inputs=(Scale,))


def test_stage_as_warm_workflow(pipeline, calls):
    # D8: the expensive part up to the cheap parameter is held, the rest reruns.
    pipeline[Filename] = 'ab'
    expected = reference(pipeline, ['ab']), reference(pipeline, ['ab'], 3.0)
    calls.reset()
    warm = Stage(pipeline, outputs=(IofQ,), inputs=(Scale,))
    assert warm.frontier == (Numerator, Denominator)
    assert warm({Scale: 1.0})[IofQ] == pytest.approx(expected[0])
    assert warm({Scale: 3.0})[IofQ] == pytest.approx(expected[1])
    assert calls['load'] == 1
    assert calls['normalize'] == 2


def test_stage_with_shared_ancestor_input_freezes_the_rest(pipeline, calls):
    # Numerator is supplied; Denominator and Scale are static and held.
    stage = Stage(pipeline, outputs=(IofQ,), inputs=(Numerator,))
    assert stage.frontier == (Denominator, Scale)
    assert stage({Numerator: [1.0]})[IofQ] == 1.0 / (sum(map(ord, 'unused')) * 4)


def test_stage_passes_through_an_output_that_is_an_input(pipeline, calls):
    stage = Stage(pipeline, outputs=(Numerator, IofQ), inputs=(Numerator,))
    assert stage.dynamic_outputs == (Numerator, IofQ)
    assert stage({Numerator: [1.0]})[Numerator] == [1.0]
    assert calls['numerator'] == 0


def test_warm_computes_shared_static_work_once(pipeline, calls):
    numerator = Stage(pipeline, outputs=(Numerator,), inputs=(Filename,))
    denominator = Stage(pipeline, outputs=(Denominator,), inputs=(Filename,))
    warm(numerator, denominator)
    assert numerator.static == {Calibration: 4.0, Bins: 2}
    assert denominator.static == {Calibration: 4.0}
    assert calls['calibration'] == 1


# --- Fold -----------------------------------------------------------------------


def test_fold_matches_manual_reduction(pipeline, calls):
    files = ['ab', 'cd', 'efg']
    expected = reference(pipeline, files)
    calls.reset()
    fold = Fold(pipeline, members=(Filename,), at=CUT, outputs=(IofQ,))
    assert fold.cut == (Numerator, Denominator)
    assert fold.compute({Filename: files})[IofQ] == pytest.approx(expected)
    assert calls['load'] == len(files)
    assert calls['calibration'] == 1
    assert calls['normalize'] == 1


def test_fold_from_dataframe_with_two_columns(pipeline):
    table = pd.DataFrame({Filename: ['ab', 'cd'], Mask: ['m', 'mmm']}).rename_axis('run')
    fold = Fold(pipeline, members=(Filename, Mask), at=CUT, outputs=(IofQ,))
    num = [97.0, 98.0, 99.0 * 3, 100.0 * 3]
    den = (97.0 + 98.0) + (99.0 + 100.0) * 3
    assert fold.compute(table)[IofQ] == pytest.approx(sum(num) / den)


def test_cut_key_independent_of_members_is_not_folded(pipeline):
    # essreflectometry wraps each reduce in try/except for this case.
    fold = Fold(pipeline, members=(Filename,), at={**CUT, Calibration: add}, outputs=(IofQ,))
    assert fold.cut == (Numerator, Denominator)
    expected = reference(pipeline, ['ab', 'cd'])
    assert fold.compute({Filename: ['ab', 'cd']})[IofQ] == pytest.approx(expected)


def test_fold_rejects_members_the_cut_does_not_need(pipeline):
    with pytest.raises(ValueError, match='not needed'):
        Fold(pipeline, members=(Filename,), at={Calibration: add}, outputs=(IofQ,))


def test_fold_three_entry_points_equal_compute(pipeline):
    # contribute, chained combine, finalize as the framework would call them (D15).
    files = ['ab', 'cd', 'efg']
    fold = Fold(pipeline, members=(Filename,), at=CUT, outputs=(IofQ,))
    partial = fold.contribute({Filename: files[0]})
    for f in files[1:]:
        partial = fold.combine([partial, fold.contribute({Filename: f})])
    assert set(partial) == {Numerator, Denominator}
    assert fold.finalize(partial)[IofQ] == pytest.approx(reference(pipeline, files))


def test_fold_without_outputs_has_no_finalize(pipeline):
    fold = Fold(pipeline, members=(Filename,), at=CUT)
    assert fold.stages == (fold.contribute_stage,)
    with pytest.raises(ValueError, match='no outputs'):
        fold.finalize(fold.contribute({Filename: 'ab'}))


def test_fold_groups_via_pandas(pipeline):
    table = pd.DataFrame({Filename: ['ab', 'cd', 'ef'], 'sample': ['x', 'y', 'x']})
    fold = Fold(pipeline, members=(Filename,), at=CUT, outputs=(IofQ,))
    results = {name: fold.compute(group[[Filename]])[IofQ] for name, group in table.groupby('sample')}
    assert results == {
        'x': pytest.approx(reference(pipeline, ['ab', 'ef'])),
        'y': pytest.approx(reference(pipeline, ['cd'])),
    }


def test_caller_holds_contributions_and_decides_what_a_parameter_change_keeps(pipeline, calls):
    # The pattern a package's notebook object implements: contributions in a dict
    # by label, a new fold when a parameter changes, contributions kept when the
    # changed key is not read by the contribute stage.
    files = ['ab', 'cd']
    expected = reference(pipeline, files, 3.0), reference(_with(pipeline, Bins, 1), files, 3.0)
    calls.reset()
    fold = Fold(pipeline, members=(Filename,), at=CUT, outputs=(IofQ,))
    held = {f: fold.contribute({Filename: f}) for f in files}

    pipeline[Scale] = 3.0
    fold = Fold(pipeline, members=(Filename,), at=CUT, outputs=(IofQ,))
    if Scale in fold.contribute_stage.keys:
        held.clear()
    held |= {f: fold.contribute({Filename: f}) for f in files if f not in held}
    assert fold.finalize(fold.combine(list(held.values())))[IofQ] == pytest.approx(expected[0])
    assert calls['load'] == 2

    pipeline[Bins] = 1
    fold = Fold(pipeline, members=(Filename,), at=CUT, outputs=(IofQ,))
    if Bins in fold.contribute_stage.keys:
        held.clear()
    held |= {f: fold.contribute({Filename: f}) for f in files if f not in held}
    assert fold.finalize(fold.combine(list(held.values())))[IofQ] == pytest.approx(expected[1])
    assert calls['load'] == 4


def test_hierarchical_fold_banks_over_runs(pipeline):
    # esssans iofq_test: banks over runs. One fold per bank on a pipeline with the
    # bank set; the per-bank results are the members of an outer fold at its own
    # cut key.
    files, masks = ['ab', 'cd'], ['m', 'mm', 'mmm']
    per_bank = [
        Fold(_with(pipeline, Mask, m), members=(Filename,), at=CUT, outputs=(IofQ,)).compute(
            {Filename: files}
        )[IofQ]
        for m in masks
    ]
    banks = Fold(pipeline, members=(IofQ,), at={IofQ: add}, outputs=(IofQ,))
    expected = sum(reference(_with(pipeline, Mask, m), files) for m in masks)
    assert banks.compute({IofQ: per_bank})[IofQ] == pytest.approx(expected)


def test_fold_is_a_snapshot_of_the_pipeline(pipeline):
    fold = Fold(pipeline, members=(Filename,), at=CUT, outputs=(IofQ,))
    pipeline[Bins] = 1
    expected = reference(_with(pipeline, Bins, 2), ['ab', 'cd'])
    assert fold.compute({Filename: ['ab', 'cd']})[IofQ] == pytest.approx(expected)


def test_compute_members(pipeline):
    per_member = compute_members(pipeline, members=(Filename,), key=IofQ, table={Filename: ['ab', 'cd']})
    assert per_member == {
        0: pytest.approx(reference(pipeline, ['ab'])),
        1: pytest.approx(reference(pipeline, ['cd'])),
    }


# --- Generics and two folds sharing a finalize -----------------------------------

SampleRun = NewType('SampleRun', int)
BackgroundRun = NewType('BackgroundRun', int)
RunType = TypeVar('RunType', SampleRun, BackgroundRun)


class File(sciline.Scope[RunType, str], str): ...


class Data(sciline.Scope[RunType, float], float): ...


Result = NewType('Result', float)


def load(f: File[RunType]) -> Data[RunType]:
    return Data[RunType](float(len(f)))


def subtract(s: Data[SampleRun], b: Data[BackgroundRun]) -> Result:
    return Result(s - b)


def test_fold_over_generic_key():
    pipeline = sciline.Pipeline([load, subtract], params={File[BackgroundRun]: 'x'})
    fold = Fold(pipeline, members=(File[SampleRun],), at={Data[SampleRun]: add}, outputs=(Result,))
    assert fold.compute({File[SampleRun]: ['ab', 'cde']})[Result] == 5 - 1


def test_two_folds_share_a_finalize_stage():
    # esssans: sample runs and background runs, each folded, one finalize stage
    # over both cuts, written by the package's object.
    pipeline = sciline.Pipeline([load, subtract])
    sample = Fold(pipeline, members=(File[SampleRun],), at={Data[SampleRun]: add})
    background = Fold(pipeline, members=(File[BackgroundRun],), at={Data[BackgroundRun]: add})
    finalize = Stage(pipeline, outputs=(Result,), inputs=sample.cut + background.cut)
    warm(sample.contribute_stage, background.contribute_stage, finalize)
    s = sample.combine([sample.contribute({File[SampleRun]: f}) for f in ['ab', 'cde']])
    b = background.combine([background.contribute({File[BackgroundRun]: f}) for f in ['x', 'yz']])
    assert finalize({**s, **b})[Result] == 5 - 3
