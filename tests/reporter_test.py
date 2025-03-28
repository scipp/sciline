# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from types import TracebackType
from typing import NewType, TypeVar

from sciline import Pipeline, Provider, Scope
from sciline.reporter import Reporter, RichReporter, TimingReporter


class RecordingReporter(Reporter):
    def __init__(self) -> None:
        super().__init__()
        self.active_providers: dict[int, Provider] = {}
        self.finished_providers: dict[int, Provider] = {}
        self.is_active = False
        self.n_steps = -1

    def __enter__(self) -> None:
        self.n_steps = self._n_steps  # copy from parent class
        self.is_active = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.is_active = False

    def on_provider_start(self, provider: Provider) -> int:
        provider_id = self._get_provider_id()
        self.active_providers[provider_id] = provider
        return provider_id

    def on_provider_end(self, provider_id: int) -> None:
        self.finished_providers[provider_id] = self.active_providers.pop(provider_id)


A = NewType('A', int)
B = NewType('B', int)
C = NewType('C', int)
D = NewType('D', int)


def f1(a: A) -> B:
    return B(a + 1)


def f2(b: B, a: A) -> C:
    return C(b * a)


def f3(b: B) -> D:
    return D(b + 2)


def merge(*x: int) -> int:
    return sum(x)


def test_reporter_records_all_function_providers() -> None:
    reporter = RecordingReporter()
    pipeline = Pipeline((f1, f2, f3), params={A: 3})
    result = pipeline.compute(C, reporter=reporter)

    assert result == 12
    assert reporter.n_steps == 2
    assert not reporter.is_active
    assert not reporter.active_providers

    finished_provider_names = sorted(
        provider.func.__name__ for provider in reporter.finished_providers.values()
    )
    assert finished_provider_names == ['f1', 'f2']  # f3 was not called


def test_reporter_records_all_function_providers_map_reduce() -> None:
    reporter = RecordingReporter()
    pipeline = Pipeline((f1, f2, f3), params={})
    pipeline[C] = pipeline[C].map({A: [2, 3]}).reduce(func=merge)
    result = pipeline.compute(C, reporter=reporter)

    assert result == 18
    assert reporter.n_steps == 5
    assert not reporter.is_active
    assert not reporter.active_providers

    finished_provider_names = sorted(
        provider.func.__name__ for provider in reporter.finished_providers.values()
    )
    assert finished_provider_names == ['f1', 'f1', 'f2', 'f2', 'merge']


T1 = NewType('T1', int)
T2 = NewType('T2', int)
T = TypeVar('T', T1, T2)


class X(Scope[T, int], int): ...


class Y(Scope[T, int], int): ...


def g1(x: X[T]) -> Y[T]:
    return Y[T](x + 10)


def g2(y1: Y[T1], y2: Y[T2], a: A) -> B:
    return B(y1 + y2 + a)


def test_reporter_records_all_function_providers_generic() -> None:
    reporter = RecordingReporter()
    pipeline = Pipeline((g1, g2, f2), params={A: 3, X[T1]: 2, X[T2]: 4})
    result = pipeline.compute(C, reporter=reporter)

    assert result == 87
    assert reporter.n_steps == 4
    assert not reporter.is_active
    assert not reporter.active_providers

    finished_provider_names = sorted(
        provider.func.__name__ for provider in reporter.finished_providers.values()
    )
    assert finished_provider_names == ['f2', 'g1', 'g1', 'g2']


def test_timing_reporter_tracks_all_calls() -> None:
    reporter = TimingReporter()
    pipeline = Pipeline((f1, f2, f3), params={})
    pipeline[C] = pipeline[C].map({A: [2, 3]}).reduce(func=merge)
    result = pipeline.compute(C, reporter=reporter)

    assert result == 18
    timings = reporter.as_pandas()
    assert len(timings) == 3
    assert (
        timings.loc[timings['Provider'] == 'tests.reporter_test.f1']['N Runs'].iloc[0]
        == 2
    )
    assert (
        timings.loc[timings['Provider'] == 'tests.reporter_test.f2']['N Runs'].iloc[0]
        == 2
    )
    assert (
        timings.loc[timings['Provider'] == 'tests.reporter_test.merge']['N Runs'].iloc[
            0
        ]
        == 1
    )


def test_rich_reporter_does_not_break_pipeline() -> None:
    # It is very difficult to test the output of RichReporter,
    # but we can at least make sure it doesn't change the final result.
    reporter = RichReporter()
    pipeline = Pipeline((f1, f2, f3), params={A: 3})
    result = pipeline.compute(C, reporter=reporter)
    assert result == 12
