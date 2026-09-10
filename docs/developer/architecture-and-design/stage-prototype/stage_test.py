from __future__ import annotations

from typing import NewType, TypeVar

import pandas as pd
import pytest
import sciline

from stage import Fold, Stage

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


def reference(pipeline: sciline.Pipeline, files: list[str], scale: float = 1.0):
    num, den = [], 0.0
    for f in files:
        p = pipeline.copy()
        p[Filename] = f
        num += p.compute(Numerator)
        den += p.compute(Denominator)
    return scale * sum(num) / den


# --- Stage ----------------------------------------------------------------------


def test_stage_computes_static_part_once_and_dynamic_part_per_call(pipeline, calls):
    stage = Stage(pipeline, outputs=(Numerator, Denominator), inputs=(Filename,))
    assert set(stage.frontier) == {Calibration, Bins}
    assert calls.counts == {}
    a = stage({Filename: 'ab'})
    b = stage({Filename: 'cd'})
    assert a[Denominator] == (97 + 98) * 4
    assert b[Denominator] == (99 + 100) * 4
    assert calls.counts == {
        'calibration': 1,
        'load': 2,
        'mask': 2,
        'numerator': 2,
        'denominator': 2,
    }


def test_stage_with_intermediate_input_cuts_its_ancestors(pipeline, calls):
    stage = Stage(pipeline, outputs=(IofQ,), inputs=(Numerator, Denominator, Scale))
    assert stage.frontier == ()
    assert calls.counts == {}
    assert stage({Numerator: [1.0, 2.0], Denominator: 4.0, Scale: 2.0})[IofQ] == 1.5
    assert calls.counts == {'normalize': 1}


def test_stage_rejects_input_the_outputs_do_not_need(pipeline):
    with pytest.raises(ValueError, match='not needed'):
        Stage(pipeline, outputs=(Denominator,), inputs=(Scale,))


def test_stage_as_warm_workflow(pipeline, calls):
    # D8: the expensive part is cached, only what depends on the cheap parameter reruns.
    pipeline[Filename] = 'abc'
    warm = Stage(pipeline, outputs=(IofQ,), inputs=(Scale,))
    first = warm({Scale: 1.0})[IofQ]
    assert calls.counts['load'] == 1
    assert warm({Scale: 3.0})[IofQ] == pytest.approx(3 * first)
    assert calls.counts['load'] == 1
    assert calls.counts['normalize'] == 2


# --- Fold -----------------------------------------------------------------------


def test_fold_matches_manual_reduction(pipeline, calls):
    files = ['ab', 'cd', 'ef']
    fold = Fold(
        pipeline,
        members={Filename: files},
        at={Numerator: concat, Denominator: add},
        outputs=(IofQ,),
    )
    assert fold.compute(IofQ) == pytest.approx(reference(pipeline, files))
    assert calls.counts['load'] == 3 + 2 * 3  # fold; reference computes twice per file
    assert calls.counts['calibration'] == 1 + 2 * 3


def test_fold_outputs_default_to_the_sinks_and_folds_compose(pipeline):
    # esssans `_set_runs` reduces its keys without naming an output, twice on one
    # pipeline (sample runs, then background runs).
    first = Fold(pipeline, members={Filename: ['ab', 'cd']}, at={Numerator: concat, Denominator: add})
    assert first.cut == (Numerator, Denominator)
    second = Fold(first.as_pipeline(), members={Mask: ['m', 'mm']}, at={IofQ: add})
    assert second.as_pipeline().compute(IofQ) == pytest.approx(
        reference(_with(pipeline, Mask, 'm'), ['ab', 'cd'])
        + reference(_with(pipeline, Mask, 'mm'), ['ab', 'cd'])
    )


def test_fold_from_dataframe_with_two_columns(pipeline):
    members = pd.DataFrame({Filename: ['ab', 'cd'], Bins: [1, 2]}).rename_axis('run')
    fold = Fold(pipeline, members=members, at={Numerator: concat, Denominator: add}, outputs=(IofQ,))
    per_member = fold.compute_members(Numerator)
    assert [len(v) for v in per_member.values()] == [1, 2]
    assert fold.compute(IofQ) == pytest.approx(
        (97 + 99 + 100) * 4 / ((97 + 98 + 99 + 100) * 4)
    )


def test_cut_key_independent_of_members_is_not_folded(pipeline):
    # essreflectometry reduces SampleRotation with `_any_value` and wraps it in
    # try/except because the key may not depend on Filename. Here it just works.
    fold = Fold(
        pipeline,
        members={Filename: ['ab', 'cd']},
        at={Numerator: concat, Denominator: add, Calibration: lambda *_: 1 / 0},
        outputs=(IofQ, Calibration),
    )
    assert fold.cut == (Numerator, Denominator)
    assert fold.compute(Calibration) == 4.0


def test_fold_three_stages_separately(pipeline):
    # D15: contribute per member, combine, finalize -- e.g. in three processes.
    files = ['ab', 'cd', 'ef']
    fold = Fold(pipeline, members={Filename: files}, at={Numerator: concat, Denominator: add}, outputs=(IofQ,))
    partial = fold.contribute(0)
    for m in (1, 2):  # chained combine, as a rule's series does
        partial = fold.combine([partial, fold.contribute(m)])
    assert fold.finalize(partial)[IofQ] == pytest.approx(reference(pipeline, files))


def test_fold_groups_via_pandas(pipeline):
    # groupby: a plain pandas groupby over the member table.
    members = pd.DataFrame({Filename: ['ab', 'cd', 'ef', 'gh'], 'angle': [1, 1, 2, 2]})
    results = {}
    for angle, group in members.groupby('angle'):
        fold = Fold(pipeline, members=group[[Filename]], at={Numerator: concat, Denominator: add}, outputs=(IofQ,))
        results[angle] = fold.compute(IofQ)
    assert results[1] == pytest.approx(reference(pipeline, ['ab', 'cd']))
    assert results[2] == pytest.approx(reference(pipeline, ['ef', 'gh']))


def test_as_pipeline_is_a_flat_pipeline_that_reruns_on_upstream_change(pipeline, calls):
    files = ['ab', 'cd']
    fold = Fold(pipeline, members={Filename: files}, at={Numerator: concat, Denominator: add}, outputs=(IofQ,))
    folded = fold.as_pipeline()
    assert Filename not in folded.underlying_graph
    assert folded.compute(IofQ) == pytest.approx(reference(pipeline, files))
    folded[Bins] = 1  # upstream of the cut: contributions rerun
    loads = calls.counts['load']
    p = pipeline.copy()
    p[Bins] = 1
    assert folded.compute(IofQ) == pytest.approx(reference(p, files))
    # Loading depends on the member alone, so it is held per member and not
    # repeated; the reference computes twice per file.
    assert calls.counts['load'] == loads + 0 + 4
    folded[Scale] = 2.0  # downstream of the cut: the same, sciline does not cache
    assert folded.compute(IofQ) == pytest.approx(2 * reference(p, files))


def test_hierarchical_fold_banks_over_runs(pipeline):
    # esssans iofq_test: banks mapped on top of run-mapped pipeline. Here: the
    # runs are folded into a flat pipeline, then folded over banks.
    files = ['ab', 'cd']
    runs = Fold(pipeline, members={Filename: files}, at={Numerator: concat, Denominator: add}, outputs=(IofQ,))
    per_run = runs.as_pipeline()
    banks = Fold(per_run, members={Mask: ['m', 'mm', 'mmm']}, at={IofQ: add}, outputs=(IofQ,))
    expected = sum(
        reference(_with(pipeline, Mask, m), files) for m in ['m', 'mm', 'mmm']
    )
    assert banks.compute(IofQ) == pytest.approx(expected)


def _with(pipeline, key, value):
    p = pipeline.copy()
    p[key] = value
    return p


# --- Generics -------------------------------------------------------------------

SampleRun = NewType('SampleRun', int)
BackgroundRun = NewType('BackgroundRun', int)
RunType = TypeVar('RunType', SampleRun, BackgroundRun)


class File(sciline.Scope[RunType, str], str): ...


class Data(sciline.Scope[RunType, float], float): ...


Result = NewType('Result', float)


def test_fold_over_generic_key():
    def load(f: File[RunType]) -> Data[RunType]:
        return Data[RunType](float(len(f)))

    def subtract(s: Data[SampleRun], b: Data[BackgroundRun]) -> Result:
        return Result(s - b)

    pipeline = sciline.Pipeline([load, subtract], params={File[BackgroundRun]: 'x'})
    fold = Fold(
        pipeline,
        members={File[SampleRun]: ['ab', 'cde']},
        at={Data[SampleRun]: add},
        outputs=(Result,),
    )
    assert fold.compute(Result) == 5 - 1


def test_stage_with_shared_ancestor_input_freezes_the_rest(pipeline, calls):
    # Numerator is supplied; Denominator and Scale are static and held.
    stage = Stage(pipeline, outputs=(IofQ,), inputs=(Numerator,))
    assert stage.frontier == (Denominator, Scale)
    assert stage({Numerator: [1.0]})[IofQ] == 1.0 / (sum(map(ord, 'unused')) * 4)


def test_fold_is_a_snapshot_of_the_pipeline(pipeline):
    fold = Fold(pipeline, members={Filename: ['ab', 'cd']}, at={Numerator: concat, Denominator: add})
    pipeline[Bins] = 1
    folded = fold.as_pipeline()
    expected = reference(_with(pipeline, Bins, 2), ['ab', 'cd'])
    assert fold.compute(IofQ) == pytest.approx(expected)
    assert folded.compute(IofQ) == pytest.approx(expected)
    assert fold.compute_members(Numerator)[0] == [97.0 * 4, 98.0 * 4]


def test_sibling_folds_on_one_pipeline_under_dask():
    # esssans `_set_runs`: sample runs and background runs folded on one pipeline,
    # computed with the default (dask) scheduler.
    from sciline.scheduler import DaskScheduler

    def total(s: Data[SampleRun], b: Data[BackgroundRun]) -> Result:
        return Result(s + b)

    def load(f: File[RunType]) -> Data[RunType]:
        return Data[RunType](float(len(f)))

    pipeline = sciline.Pipeline([load, total])
    sample = Fold(pipeline, members={File[SampleRun]: ['a', 'bb']}, at={Data[SampleRun]: add})
    both = Fold(sample.as_pipeline(), members={File[BackgroundRun]: ['ccc']}, at={Data[BackgroundRun]: add})
    assert both.as_pipeline().compute(Result, scheduler=DaskScheduler()) == 6
