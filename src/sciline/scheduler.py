# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

Key = type
Graph = Dict[
    Key,
    Tuple[Callable[..., Any], Dict[str, Key]],
]


class Scheduler(Protocol):
    def get(self, graph: Graph, keys: List[type]) -> Any:
        ...


class NaiveScheduler:
    """
    A naive scheduler that computes intermediate results and results in order.

    May consume excessive memory since intermediate results are not freed eagerly,
    but kept until returning the final result. Prefer installing `dask` and using
    :py:class:`DaskScheduler` instead.
    """

    def get(self, graph: Graph, keys: List[type]) -> Any:
        from graphlib import TopologicalSorter

        results: Dict[type, Any] = {}
        dependencies = {tp: set(args.values()) for tp, (_, args) in graph.items()}
        ts = TopologicalSorter(dependencies)
        for t in ts.static_order():
            provider, args = graph[t]
            args = {name: results[arg] for name, arg in args.items()}
            results[t] = provider(*args.values())
        return tuple(results[key] for key in keys)


class DaskScheduler:
    def __init__(self, scheduler: Optional[Callable[..., Any]] = None) -> None:
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

    def get(self, graph: Graph, keys: List[type]) -> Any:
        dsk = {tp: (provider, *args.values()) for tp, (provider, args) in graph.items()}
        return self._dask_get(dsk, keys)
