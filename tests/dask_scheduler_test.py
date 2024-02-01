# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

import sciline as sl

dask = pytest.importorskip("dask")


def test_dask_scheduler_is_threaded_by_default() -> None:
    scheduler = sl.scheduler.DaskScheduler()
    assert scheduler._dask_get == dask.threaded.get


def test_pipeline_simple_provider_and_param_with_dask() -> None:
    def fn(x: int) -> str:
        return f'x = {x}'

    scheduler = sl.scheduler.DaskScheduler()
    pipeline = sl.Pipeline([fn], params={int: 3})
    assert pipeline.compute(str, scheduler=scheduler) == 'x = 3'


def test_pipeline_keyword_argument_and_param_with_dask() -> None:
    def fn_with_kwarg(*, y: int) -> str:
        return f'y = {y}'

    scheduler = sl.scheduler.DaskScheduler()
    pipeline = sl.Pipeline([fn_with_kwarg], params={int: 5})
    assert pipeline.compute(str, scheduler=scheduler) == 'y = 5'


def test_pipeline_mixed_arguments_with_dask() -> None:
    def no_args() -> float:
        return 1.2

    def pos_only(a: float) -> list:
        return [a, a]

    def kwarg_only(*, lst: list) -> int:
        return len(lst)

    def mixed_args(i: int, *, lst: list) -> str:
        return f'i = {i}, lst[0] = {lst[0]}'

    scheduler = sl.scheduler.DaskScheduler()
    pipeline = sl.Pipeline([no_args, pos_only, kwarg_only, mixed_args])
    assert pipeline.compute(str, scheduler=scheduler) == 'i = 2, lst[0] = 1.2'
