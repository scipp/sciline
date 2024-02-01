# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

import sciline as sl

dask = pytest.importorskip("dask")


def test_dask_scheduler_is_threaded_by_default() -> None:
    scheduler = sl.scheduler.DaskScheduler()
    assert scheduler._dask_get == dask.threaded.get
