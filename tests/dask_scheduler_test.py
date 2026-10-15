# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable
from typing import Any, Generic, NewType, TypeVar

import pytest
from pydantic import BaseModel

import sciline as sl

dask = pytest.importorskip("dask")


def test_dask_scheduler_is_threaded_by_default() -> None:
    scheduler = sl.scheduler.DaskScheduler()
    assert scheduler._dask_get == dask.threaded.get


@pytest.mark.parametrize('dask_get', [dask.get, dask.threaded.get])
def test_dask_scheduler_preserves_distinct_keys_with_equal_names(
    dask_get: Callable[..., Any],
) -> None:
    class First(str):
        pass

    class Second(str):
        pass

    Second.__qualname__ = First.__qualname__

    def combine(first: First, *, second: Second) -> str:
        return f'{first}:{second}'

    pipeline = sl.Pipeline(
        [combine], params={First: First('first'), Second: Second('second')}
    )

    assert pipeline.compute(
        (First, Second, str), scheduler=sl.scheduler.DaskScheduler(dask_get)
    ) == {First: 'first', Second: 'second', str: 'first:second'}


SampleRun = NewType('SampleRun', int)
CanRun = NewType('CanRun', int)
RunType = TypeVar('RunType', SampleRun, CanRun)


class Run(BaseModel, Generic[RunType]):
    file: str


class Filename(sl.Scope[RunType, str], str):
    pass


def filename(run: Run[RunType]) -> Filename[RunType]:
    return Filename[RunType](run.file)


@pytest.mark.parametrize('dask_get', [dask.get, dask.threaded.get])
def test_dask_scheduler_preserves_generic_models_over_new_types(
    dask_get: Callable[..., Any],
) -> None:
    pipeline = sl.Pipeline([filename])
    pipeline[Run[SampleRun]] = Run[SampleRun](file='sample.nxs')
    pipeline[Run[CanRun]] = Run[CanRun](file='can.nxs')

    sample, can = Filename[SampleRun], Filename[CanRun]
    assert pipeline.compute(
        (sample, can), scheduler=sl.scheduler.DaskScheduler(dask_get)
    ) == {sample: 'sample.nxs', can: 'can.nxs'}
