# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

Key = type
Graph = Dict[
    Key,
    Tuple[Callable[..., Any], Dict[str, Key]],
]


class Scheduler(Protocol):
    def get(self, graph: Graph, keys: List[type]):
        ...


class DaskScheduler:
    def __init__(self, scheduler: Optional[Callable[..., Any]] = None):
        """Wrapper for a Dask scheduler.

        Note that this currently only works if all providers support posargs.

        Parameters
        ----------
        scheduler:
            A Dask scheduler, such as `dask.get`, `dask.threaded.get`,
            `dask.multiprocessing.get, or `dask.distributed.Client.get`.
        """
        if scheduler is None:
            import dask

            self._dask_get = dask.get
        else:
            self._dask_get = scheduler

    def get(self, graph: Graph, keys: List[type]):
        dsk = {tp: (provider, *args.values()) for tp, (provider, args) in graph.items()}
        return self._dask_get(dsk, keys)
