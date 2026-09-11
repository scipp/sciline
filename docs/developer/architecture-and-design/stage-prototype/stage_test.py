from __future__ import annotations

from typing import Any, NewType, TypeVar

import pandas as pd
import pytest
from stage import Aggregation, Buffered, Stage, compute_members, warm

import sciline

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
Scale = NewType('Scale', float)  # cheap parameter, after the accumulation keys


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


ACCUMULATORS = {Numerator: Buffered(concat), Denominator: Buffered(add)}


def files_table(files: list[str]) -> dict[int, dict[Any, str]]:
    return {i: {Filename: f} for i, f in enumerate(files)}


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


# --- Accumulators ---------------------------------------------------------------


def test_buffered_makes_fresh_accumulators_that_apply_func_in_push_order():
    def join(*parts: str) -> str:
        return ''.join(parts)

    make = Buffered(join)
    a, b = make(), make()
    a.push('x')
    a.push('y')
    b.push('z')
    assert a.value == 'xy'
    assert b.value == 'z'


def test_buffered_accumulator_without_pushes_has_no_value():
    acc = Buffered(add)()
    with pytest.raises(ValueError, match='Nothing has been pushed'):
        _ = acc.value


class RunningSum:
    """An accumulator holding a running total, never the pushed values."""

    def __init__(self) -> None:
        self.value = 0.0

    def push(self, value: float) -> None:
        self.value += value


class RunningConcat:
    def __init__(self) -> None:
        self.value: list[float] = []

    def push(self, value: list[float]) -> None:
        self.value = self.value + value


def test_custom_accumulator_class_as_factory(pipeline):
    files = ['ab', 'cd', 'efg']
    agg = Aggregation(
        pipeline,
        members=(Filename,),
        accumulators={Numerator: RunningConcat, Denominator: RunningSum},
        outputs=(IofQ,),
    )
    expected = reference(pipeline, files)
    assert agg.compute(files_table(files))[IofQ] == pytest.approx(expected)


Value = NewType('Value', int)
Total = NewType('Total', int)


def test_compute_pushes_each_contribution_before_the_next_is_made():
    log: list[str] = []

    def contribute(value: Value) -> Total:
        log.append('contribute')
        return Total(value)

    class LoggingSum:
        def __init__(self) -> None:
            self.value = 0

        def push(self, value: int) -> None:
            log.append('push')
            self.value += value

    pipeline = sciline.Pipeline([contribute])
    agg = Aggregation(
        pipeline, members=(Value,), accumulators={Total: LoggingSum}, outputs=(Total,)
    )
    table = {i: {Value: v} for i, v in enumerate([1, 2, 3])}
    assert agg.compute(table)[Total] == 6
    assert log == ['contribute', 'push'] * 3


# --- Aggregation ----------------------------------------------------------------


def test_aggregation_matches_manual_reduction(pipeline, calls):
    files = ['ab', 'cd', 'efg']
    expected = reference(pipeline, files)
    calls.reset()
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    assert agg.accumulation_keys == (Numerator, Denominator)
    assert agg.compute(files_table(files))[IofQ] == pytest.approx(expected)
    assert calls['load'] == len(files)
    assert calls['calibration'] == 1
    assert calls['normalize'] == 1


def test_aggregation_over_table_with_two_columns(pipeline):
    table = {
        'run1': {Filename: 'ab', Mask: 'm'},
        'run2': {Filename: 'cd', Mask: 'mmm'},
    }
    agg = Aggregation(
        pipeline, members=(Filename, Mask), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    num = [97.0, 98.0, 99.0 * 3, 100.0 * 3]
    den = (97.0 + 98.0) + (99.0 + 100.0) * 3
    assert agg.compute(table)[IofQ] == pytest.approx(sum(num) / den)


def test_key_independent_of_members_is_not_accumulated(pipeline):
    # essreflectometry wraps each reduce in try/except for this case.
    agg = Aggregation(
        pipeline,
        members=(Filename,),
        accumulators={**ACCUMULATORS, Calibration: Buffered(add)},
        outputs=(IofQ,),
    )
    assert agg.accumulation_keys == (Numerator, Denominator)
    assert set(agg.accumulators()) == {Numerator, Denominator}
    expected = reference(pipeline, ['ab', 'cd'])
    assert agg.compute(files_table(['ab', 'cd']))[IofQ] == pytest.approx(expected)


def test_aggregation_rejects_members_the_accumulation_keys_do_not_need(pipeline):
    with pytest.raises(ValueError, match='not needed'):
        Aggregation(
            pipeline,
            members=(Filename,),
            accumulators={Calibration: Buffered(add)},
            outputs=(IofQ,),
        )


def test_aggregation_three_entry_points_equal_compute(pipeline):
    # contribute, chained combine, finalize as the framework would call them (D15).
    files = ['ab', 'cd', 'efg']
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    partial = agg.contribute({Filename: files[0]})
    for f in files[1:]:
        partial = agg.combine([partial, agg.contribute({Filename: f})])
    assert set(partial) == {Numerator, Denominator}
    assert agg.finalize(partial)[IofQ] == pytest.approx(reference(pipeline, files))


def test_aggregation_without_outputs_has_no_finalize(pipeline):
    agg = Aggregation(pipeline, members=(Filename,), accumulators=ACCUMULATORS)
    assert agg.stages == (agg.contribute_stage,)
    with pytest.raises(ValueError, match='no outputs'):
        agg.finalize(agg.contribute({Filename: 'ab'}))


def test_aggregation_groups_via_pandas(pipeline):
    table = pd.DataFrame({Filename: ['ab', 'cd', 'ef'], 'sample': ['x', 'y', 'x']})
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    results = {
        name: agg.compute(group[[Filename]].to_dict('index'))[IofQ]
        for name, group in table.groupby('sample')
    }
    assert results == {
        'x': pytest.approx(reference(pipeline, ['ab', 'ef'])),
        'y': pytest.approx(reference(pipeline, ['cd'])),
    }


def test_caller_holds_contributions_and_decides_what_a_parameter_change_keeps(
    pipeline, calls
):
    # The pattern a package's notebook object implements: contributions in a dict
    # by label, a new aggregation when a parameter changes, contributions kept when
    # the changed key is not read by the contribute stage.
    files = ['ab', 'cd']
    expected = (
        reference(pipeline, files, 3.0),
        reference(_with(pipeline, Bins, 1), files, 3.0),
    )
    calls.reset()
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    held = {f: agg.contribute({Filename: f}) for f in files}

    pipeline[Scale] = 3.0
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    if Scale in agg.contribute_stage.keys:
        held.clear()
    held |= {f: agg.contribute({Filename: f}) for f in files if f not in held}
    assert agg.finalize(agg.combine(held.values()))[IofQ] == pytest.approx(expected[0])
    assert calls['load'] == 2

    pipeline[Bins] = 1
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    if Bins in agg.contribute_stage.keys:
        held.clear()
    held |= {f: agg.contribute({Filename: f}) for f in files if f not in held}
    assert agg.finalize(agg.combine(held.values()))[IofQ] == pytest.approx(expected[1])
    assert calls['load'] == 4


def test_hierarchical_aggregation_banks_over_runs(pipeline):
    # esssans iofq_test: banks over runs. One aggregation per bank on a pipeline
    # with the bank set; the per-bank results are the members of an outer
    # aggregation at its own accumulation key.
    files, masks = ['ab', 'cd'], ['m', 'mm', 'mmm']
    per_bank = {
        m: Aggregation(
            _with(pipeline, Mask, m),
            members=(Filename,),
            accumulators=ACCUMULATORS,
            outputs=(IofQ,),
        ).compute(files_table(files))[IofQ]
        for m in masks
    }
    banks = Aggregation(
        pipeline, members=(IofQ,), accumulators={IofQ: Buffered(add)}, outputs=(IofQ,)
    )
    expected = sum(reference(_with(pipeline, Mask, m), files) for m in masks)
    table = {m: {IofQ: v} for m, v in per_bank.items()}
    assert banks.compute(table)[IofQ] == pytest.approx(expected)


def test_aggregation_is_a_snapshot_of_the_pipeline_parameters(pipeline):
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    pipeline[Bins] = 1
    expected = reference(_with(pipeline, Bins, 2), ['ab', 'cd'])
    assert agg.compute(files_table(['ab', 'cd']))[IofQ] == pytest.approx(expected)


def doubled_denominator(data: Masked) -> Denominator:
    return Denominator(2 * sum(data))


def test_aggregation_is_a_snapshot_of_the_pipeline_graph(pipeline):
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    expected = reference(pipeline, ['ab', 'cd'])

    other = pipeline.copy()
    other.insert(doubled_denominator)
    pipeline[Denominator] = other[Denominator]
    assert reference(pipeline, ['ab', 'cd']) == pytest.approx(expected / 2)
    assert agg.compute(files_table(['ab', 'cd']))[IofQ] == pytest.approx(expected)


def test_compute_members(pipeline):
    per_member = compute_members(
        pipeline, members=(Filename,), key=IofQ, table=files_table(['ab', 'cd'])
    )
    assert per_member == {
        0: pytest.approx(reference(pipeline, ['ab'])),
        1: pytest.approx(reference(pipeline, ['cd'])),
    }


# --- Generics and two aggregations sharing a finalize ----------------------------

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


def test_aggregation_over_generic_key():
    pipeline = sciline.Pipeline([load, subtract], params={File[BackgroundRun]: 'x'})
    agg = Aggregation(
        pipeline,
        members=(File[SampleRun],),
        accumulators={Data[SampleRun]: Buffered(add)},
        outputs=(Result,),
    )
    table = {i: {File[SampleRun]: f} for i, f in enumerate(['ab', 'cde'])}
    assert agg.compute(table)[Result] == 5 - 1


def test_two_aggregations_share_a_finalize_stage():
    # esssans: sample runs and background runs, each aggregated, one finalize stage
    # over both sets of accumulation keys, written by the package's object.
    pipeline = sciline.Pipeline([load, subtract])
    sample = Aggregation(
        pipeline,
        members=(File[SampleRun],),
        accumulators={Data[SampleRun]: Buffered(add)},
    )
    background = Aggregation(
        pipeline,
        members=(File[BackgroundRun],),
        accumulators={Data[BackgroundRun]: Buffered(add)},
    )
    finalize = Stage(
        pipeline,
        outputs=(Result,),
        inputs=sample.accumulation_keys + background.accumulation_keys,
    )
    warm(sample.contribute_stage, background.contribute_stage, finalize)
    s = sample.combine(sample.contribute({File[SampleRun]: f}) for f in ['ab', 'cde'])
    b = background.combine(
        background.contribute({File[BackgroundRun]: f}) for f in ['x', 'yz']
    )
    assert finalize({**s, **b})[Result] == 5 - 3


def test_compute_rejects_an_empty_table(pipeline):
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    with pytest.raises(ValueError, match='empty'):
        agg.compute({})
