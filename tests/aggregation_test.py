# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from typing import Any, NewType, TypeVar

import pandas as pd
import pytest

import sciline as sl
from sciline import Aggregation, Buffered, Stage, compute_members, warm
from sciline.aggregation import Table
from sciline.typing import Key

Filename = NewType('Filename', str)
Mask = NewType('Mask', str)
Calibration = NewType('Calibration', float)  # static, expensive
Loaded = NewType('Loaded', list[float])  # per file
Masked = NewType('Masked', list[float])
Bins = NewType('Bins', int)
Numerator = NewType('Numerator', list[float])  # per file, combined by concat
Denominator = NewType('Denominator', float)  # per file, combined by sum
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


def concat(*parts: list[float]) -> list[float]:
    return [x for part in parts for x in part]


def add(*parts: float) -> float:
    return sum(parts)


ACCUMULATORS: dict[Key, Buffered[Any]] = {
    Numerator: Buffered(concat),
    Denominator: Buffered(add),
}


def files_table(files: list[str]) -> Table:
    return {i: {Filename: f} for i, f in enumerate(files)}


def reference(pipeline: sl.Pipeline, files: list[str], scale: float = 1.0) -> float:
    num: list[float] = []
    den = 0.0
    for f in files:
        p = pipeline.copy()
        p[Filename] = f
        num += p.compute(Numerator)
        den += p.compute(Denominator)
    return scale * sum(num) / den


def with_param(pipeline: sl.Pipeline, key: Key, value: Any) -> sl.Pipeline:
    p = pipeline.copy()
    p[key] = value
    return p


def test_buffered_makes_fresh_accumulators_that_apply_func_in_push_order() -> None:
    def join(*parts: str) -> str:
        return ''.join(parts)

    make = Buffered(join)
    a, b = make(), make()
    a.push('x')
    a.push('y')
    b.push('z')
    assert a.value == 'xy'
    assert b.value == 'z'


def test_buffered_accumulator_without_pushes_has_no_value() -> None:
    acc = Buffered(add)()
    with pytest.raises(ValueError, match='Nothing has been pushed'):
        acc.value


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


def test_custom_accumulator_class_as_factory(pipeline: sl.Pipeline) -> None:
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


def test_compute_pushes_each_contribution_before_the_next_is_made() -> None:
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

    pipeline = sl.Pipeline([contribute])
    agg = Aggregation(
        pipeline, members=(Value,), accumulators={Total: LoggingSum}, outputs=(Total,)
    )
    table: Table = {i: {Value: v} for i, v in enumerate([1, 2, 3])}
    assert agg.compute(table)[Total] == 6
    assert log == ['contribute', 'push'] * 3


def test_aggregation_matches_manual_reduction(
    pipeline: sl.Pipeline, calls: Calls
) -> None:
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


def test_aggregation_over_table_with_two_columns(pipeline: sl.Pipeline) -> None:
    table: Table = {
        'run1': {Filename: 'ab', Mask: 'm'},
        'run2': {Filename: 'cd', Mask: 'mmm'},
    }
    agg = Aggregation(
        pipeline, members=(Filename, Mask), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    num = [97.0, 98.0, 99.0 * 3, 100.0 * 3]
    den = (97.0 + 98.0) + (99.0 + 100.0) * 3
    assert agg.compute(table)[IofQ] == pytest.approx(sum(num) / den)


def test_key_independent_of_members_is_not_accumulated(pipeline: sl.Pipeline) -> None:
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


def test_aggregation_rejects_members_the_accumulation_keys_do_not_need(
    pipeline: sl.Pipeline,
) -> None:
    with pytest.raises(ValueError, match='not needed'):
        Aggregation(
            pipeline,
            members=(Filename,),
            accumulators={Calibration: Buffered(add)},
            outputs=(IofQ,),
        )


def test_aggregation_three_entry_points_equal_compute(pipeline: sl.Pipeline) -> None:
    files = ['ab', 'cd', 'efg']
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    partial = agg.contribute({Filename: files[0]})
    for f in files[1:]:
        partial = agg.combine([partial, agg.contribute({Filename: f})])
    assert set(partial) == {Numerator, Denominator}
    assert agg.finalize(partial)[IofQ] == pytest.approx(reference(pipeline, files))


def test_aggregation_without_outputs_has_no_finalize(pipeline: sl.Pipeline) -> None:
    agg = Aggregation(pipeline, members=(Filename,), accumulators=ACCUMULATORS)
    assert agg.stages == (agg.contribute_stage,)
    with pytest.raises(ValueError, match='no outputs'):
        agg.finalize(agg.contribute({Filename: 'ab'}))


def test_aggregation_groups_via_pandas(pipeline: sl.Pipeline) -> None:
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
    pipeline: sl.Pipeline, calls: Calls
) -> None:
    # Contributions held by label; a new aggregation when a parameter changes;
    # contributions kept when the changed key is not read by the contribute stage.
    files = ['ab', 'cd']
    expected = (
        reference(pipeline, files, 3.0),
        reference(with_param(pipeline, Bins, 1), files, 3.0),
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


def test_hierarchical_aggregation_banks_over_runs(pipeline: sl.Pipeline) -> None:
    # One aggregation per bank on a pipeline with the bank set; the per-bank
    # results are the members of an outer aggregation at its own accumulation key.
    files, masks = ['ab', 'cd'], ['m', 'mm', 'mmm']
    per_bank = {
        m: Aggregation(
            with_param(pipeline, Mask, m),
            members=(Filename,),
            accumulators=ACCUMULATORS,
            outputs=(IofQ,),
        ).compute(files_table(files))[IofQ]
        for m in masks
    }
    banks = Aggregation(
        pipeline, members=(IofQ,), accumulators={IofQ: Buffered(add)}, outputs=(IofQ,)
    )
    expected = sum(reference(with_param(pipeline, Mask, m), files) for m in masks)
    table: Table = {m: {IofQ: v} for m, v in per_bank.items()}
    assert banks.compute(table)[IofQ] == pytest.approx(expected)


def test_aggregation_is_a_snapshot_of_the_pipeline_parameters(
    pipeline: sl.Pipeline,
) -> None:
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    pipeline[Bins] = 1
    expected = reference(with_param(pipeline, Bins, 2), ['ab', 'cd'])
    assert agg.compute(files_table(['ab', 'cd']))[IofQ] == pytest.approx(expected)


def doubled_denominator(data: Masked) -> Denominator:
    return Denominator(2 * sum(data))


def test_aggregation_is_a_snapshot_of_the_pipeline_graph(pipeline: sl.Pipeline) -> None:
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    expected = reference(pipeline, ['ab', 'cd'])

    other = pipeline.copy()
    other.insert(doubled_denominator)
    pipeline[Denominator] = other[Denominator]
    assert reference(pipeline, ['ab', 'cd']) == pytest.approx(expected / 2)
    assert agg.compute(files_table(['ab', 'cd']))[IofQ] == pytest.approx(expected)


def test_compute_rejects_an_empty_table(pipeline: sl.Pipeline) -> None:
    agg = Aggregation(
        pipeline, members=(Filename,), accumulators=ACCUMULATORS, outputs=(IofQ,)
    )
    with pytest.raises(ValueError, match='empty'):
        agg.compute({})


def test_compute_members(pipeline: sl.Pipeline) -> None:
    per_member = compute_members(
        pipeline, members=(Filename,), key=IofQ, table=files_table(['ab', 'cd'])
    )
    assert per_member == {
        0: pytest.approx(reference(pipeline, ['ab'])),
        1: pytest.approx(reference(pipeline, ['cd'])),
    }


SampleRun = NewType('SampleRun', int)
BackgroundRun = NewType('BackgroundRun', int)
RunType = TypeVar('RunType', SampleRun, BackgroundRun)


class File(sl.Scope[RunType, str], str): ...


class Data(sl.Scope[RunType, float], float): ...


Result = NewType('Result', float)


def load(f: File[RunType]) -> Data[RunType]:
    return Data[RunType](float(len(f)))  # type: ignore[return-value]


def subtract(s: Data[SampleRun], b: Data[BackgroundRun]) -> Result:
    return Result(s - b)


def test_aggregation_over_generic_key() -> None:
    pipeline = sl.Pipeline([load, subtract], params={File[BackgroundRun]: 'x'})
    agg = Aggregation(
        pipeline,
        members=(File[SampleRun],),
        accumulators={Data[SampleRun]: Buffered(add)},
        outputs=(Result,),
    )
    table: Table = {i: {File[SampleRun]: f} for i, f in enumerate(['ab', 'cde'])}
    assert agg.compute(table)[Result] == 5 - 1


def test_two_aggregations_share_a_finalize_stage() -> None:
    # Sample runs and background runs, each aggregated, one finalize stage over
    # both sets of accumulation keys.
    pipeline = sl.Pipeline([load, subtract])
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
