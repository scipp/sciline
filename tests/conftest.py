# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Optional

import pytest

from sciline.scheduler import DaskScheduler, NaiveScheduler, Scheduler


@pytest.fixture
def naive_scheduler() -> NaiveScheduler:
    return NaiveScheduler()


@pytest.fixture
def dask_scheduler() -> Optional[DaskScheduler]:
    try:
        import dask  # noqa: F401

        return DaskScheduler()
    except ModuleNotFoundError:
        return None


@pytest.fixture(params=['naive', 'dask'])
def scheduler(request: pytest.FixtureRequest) -> Scheduler:
    if request.param == 'naive':
        return request.getfixturevalue('naive_scheduler')  # type: ignore[no-any-return]

    sched = request.getfixturevalue('dask_scheduler')
    if sched is None:
        pytest.skip("Test requires Dask")
    return sched  # type: ignore[no-any-return]
