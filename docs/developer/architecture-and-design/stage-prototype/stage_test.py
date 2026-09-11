from __future__ import annotations

from typing import NewType, TypeVar

import pandas as pd
import pytest
import sciline

from stage import Fold, Stage, warm

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
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
    fold.set_members({Filename: files})
    assert fold.cut == (Numerator, Denominator)
    assert fold.compute(IofQ) == pytest.approx(expected)
    assert calls['load'] == len(files)
    assert calls['calibration'] == 1
    assert calls['normalize'] == 1


def test_fold_outputs_default_to_the_sinks(pipeline):
    fold = Fold(pipeline, over={Filename: CUT})
    fold.set_members({Filename: ['ab', 'cd']})
    assert fold.compute() == {IofQ: pytest.approx(reference(pipeline, ['ab', 'cd']))}


def test_fold_from_dataframe_with_two_columns(pipeline):
    members = pd.DataFrame({Filename: ['ab', 'cd'], Mask: ['m', 'mmm']}).rename_axis('run')
    fold = Fold(pipeline, over={(Filename, Mask): CUT}, outputs=(IofQ,))
    fold.set_members(members)
    num = [97.0, 98.0, 99.0 * 3, 100.0 * 3]
    den = (97.0 + 98.0) + (99.0 + 100.0) * 3
    assert fold.compute(IofQ) == pytest.approx(sum(num) / den)
    assert list(fold.members[(Filename, Mask)]) == [0, 1]


def test_cut_key_independent_of_members_is_not_folded(pipeline):
    # essreflectometry wraps each reduce in try/except for this case.
    fold = Fold(pipeline, over={Filename: {**CUT, Calibration: add}}, outputs=(IofQ,))
    fold.set_members({Filename: ['ab', 'cd']})
    assert fold.cut == (Numerator, Denominator)
    assert fold.compute(IofQ) == pytest.approx(reference(pipeline, ['ab', 'cd']))


def test_fold_rejects_a_group_whose_cut_keys_are_all_static(pipeline):
    fold = Fold(pipeline, over={Filename: {Calibration: add}}, outputs=(IofQ,))
    fold.set_members({Filename: ['ab']})
    with pytest.raises(ValueError, match='not needed'):
        fold.compute()


def test_fold_three_entry_points_equal_compute(pipeline):
    # contribute, chained combine, finalize as the framework would call them (D15).
    files = ['ab', 'cd', 'efg']
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
    partial = fold.contribute({Filename: files[0]})
    for f in files[1:]:
        partial = fold.combine([partial, fold.contribute({Filename: f})])
    assert set(partial) == {Numerator, Denominator}
    assert fold.finalize([partial])[IofQ] == pytest.approx(reference(pipeline, files))


def test_fold_groups_via_pandas(pipeline):
    table = pd.DataFrame({Filename: ['ab', 'cd', 'ef'], 'sample': ['x', 'y', 'x']})
    results = {}
    for name, group in table.groupby('sample'):
        fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
        fold.set_members(group[[Filename]])
        results[name] = fold.compute(IofQ)
    assert results == {
        'x': pytest.approx(reference(pipeline, ['ab', 'ef'])),
        'y': pytest.approx(reference(pipeline, ['cd'])),
    }


def test_adding_a_member_costs_one_contribution(pipeline, calls):
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
    fold.set_members({Filename: ['ab', 'cd']})
    fold.compute(IofQ)
    fold.set_members({Filename: ['ab', 'cd', 'efg']})
    result = fold.compute(IofQ)
    assert calls['load'] == 3
    assert calls['normalize'] == 2
    assert result == pytest.approx(reference(pipeline, ['ab', 'cd', 'efg']))


def test_replacing_a_member_drops_its_contribution(pipeline, calls):
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
    fold.set_members({Filename: ['ab', 'cd']})
    fold.compute(IofQ)
    fold.set_members({Filename: ['ab', 'xy']})
    result = fold.compute(IofQ)
    assert calls['load'] == 3
    assert result == pytest.approx(reference(pipeline, ['ab', 'xy']))


def test_parameter_after_the_cut_keeps_contributions(pipeline, calls):
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
    fold.set_members({Filename: ['ab', 'cd']})
    fold.compute(IofQ)
    fold[Scale] = 3.0
    result = fold.compute(IofQ)
    assert calls['load'] == 2
    assert calls['numerator'] == 2
    assert calls['normalize'] == 2
    assert result == pytest.approx(reference(pipeline, ['ab', 'cd'], 3.0))


def test_parameter_before_the_cut_drops_contributions_but_keeps_members(pipeline, calls):
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,), keep_members=True)
    fold.set_members({Filename: ['ab', 'cd']})
    fold.compute(IofQ)
    fold[Bins] = 1
    result = fold.compute(IofQ)
    assert calls['load'] == 2
    assert calls['numerator'] == 4
    assert result == pytest.approx(reference(_with(pipeline, Bins, 1), ['ab', 'cd']))


def test_setting_a_member_key_as_parameter_is_refused(pipeline):
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
    with pytest.raises(ValueError, match='member key'):
        fold[Filename] = 'ab'


def test_hierarchical_fold_banks_over_runs(pipeline, calls):
    # esssans iofq_test: banks over runs. The inner fold over runs is reused per
    # bank with the bank set as a parameter; the loaded runs are held. The outer
    # fold's members are the inner results, at the outer cut key itself.
    files, masks = ['ab', 'cd'], ['m', 'mm', 'mmm']
    expected = sum(reference(_with(pipeline, Mask, m), files) for m in masks)
    calls.reset()
    runs = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,), keep_members=True)
    runs.set_members({Filename: files})
    per_bank = []
    for mask in masks:
        runs[Mask] = mask
        per_bank.append(runs.compute(IofQ))
    banks = Fold(pipeline, over={IofQ: {IofQ: add}}, outputs=(IofQ,))
    banks.set_members({IofQ: per_bank})
    assert banks.compute(IofQ) == pytest.approx(expected)
    assert calls['load'] == len(files)
    assert calls['mask'] == len(files) * len(masks)


def test_fold_is_a_snapshot_of_the_pipeline(pipeline):
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
    fold.set_members({Filename: ['ab', 'cd']})
    pipeline[Bins] = 1
    assert fold.compute(IofQ) == pytest.approx(reference(_with(pipeline, Bins, 2), ['ab', 'cd']))
    assert fold.compute_members(Numerator)[0] == [97.0 * 4, 98.0 * 4]


def test_compute_members_of_a_key_after_the_cut(pipeline):
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
    fold.set_members({Filename: ['ab', 'cd']})
    per_member = fold.compute_members(IofQ)
    assert per_member == {
        0: pytest.approx(reference(pipeline, ['ab'])),
        1: pytest.approx(reference(pipeline, ['cd'])),
    }


def test_table_with_unknown_columns_is_refused(pipeline):
    fold = Fold(pipeline, over={Filename: CUT}, outputs=(IofQ,))
    with pytest.raises(KeyError, match='No group'):
        fold.set_members({Mask: ['m']})


# --- Generics and several groups -------------------------------------------------

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
    fold = Fold(pipeline, over={File[SampleRun]: {Data[SampleRun]: add}}, outputs=(Result,))
    fold.set_members({File[SampleRun]: ['ab', 'cde']})
    assert fold.compute(Result) == 5 - 1


def test_two_groups_on_one_pipeline():
    # esssans: sample runs and background runs, each folded, one finalize.
    pipeline = sciline.Pipeline([load, subtract])
    fold = Fold(
        pipeline,
        over={
            File[SampleRun]: {Data[SampleRun]: add},
            File[BackgroundRun]: {Data[BackgroundRun]: add},
        },
        outputs=(Result,),
    )
    fold.set_members({File[SampleRun]: ['ab', 'cde']})
    with pytest.raises(ValueError, match='No members'):
        fold.compute()
    fold.set_members({File[BackgroundRun]: ['x', 'yz']})
    assert fold.compute(Result) == 5 - 3
    sample = fold.combine([fold.contribute({File[SampleRun]: f}) for f in ['a', 'b']])
    background = fold.contribute({File[BackgroundRun]: 'xyz'})
    assert fold.finalize([sample, background])[Result] == 2 - 3
