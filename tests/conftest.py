# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any

import pytest

from sciline.scheduler import DaskScheduler, NaiveScheduler, Scheduler


@pytest.fixture
def naive_scheduler() -> NaiveScheduler:
    return NaiveScheduler()


@pytest.fixture
def dask_scheduler() -> DaskScheduler | None:
    try:
        import dask  # noqa: F401

        return DaskScheduler()
    except ModuleNotFoundError:
        pytest.skip("Test requires Dask")


@pytest.fixture(scope='session')
def dask_distributed_client() -> Any:
    """Manage a singleton Dask client for all tests.

    Tests must always use this client if they need to use dask.distributed.
    They must never create their own client.
    Otherwise, the clients will be in conflict.
    """
    try:
        from dask.distributed import Client

        return Client(set_as_default=False)
    except ImportError:
        pytest.skip("Test requires dask.distributed")


@pytest.fixture
def dask_distributed_scheduler(dask_distributed_client: Any) -> DaskScheduler | None:
    if dask_distributed_client is None:
        return None
    return DaskScheduler(dask_distributed_client.get)


@pytest.fixture(params=['naive', 'dask', 'dask_distributed'])
def scheduler(request: pytest.FixtureRequest) -> Scheduler:
    match request.param:
        case 'naive':
            return request.getfixturevalue('naive_scheduler')  # type: ignore[no-any-return]
        case 'dask':
            return request.getfixturevalue('dask_scheduler')  # type: ignore[no-any-return]
        case 'dask_distributed':
            return request.getfixturevalue('dask_distributed_scheduler')  # type: ignore[no-any-return]
        case _:
            raise ValueError(f"Unknown scheduler: {request.param}")
