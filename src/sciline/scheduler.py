# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

Key = type
Graph = Dict[
    Key,
    Tuple[Callable[..., Any], Dict[str, Key]],
]


class CycleError(Exception):
    pass


class Scheduler(Protocol):
    """
    Scheduler interface compatible with :py:class:`sciline.Pipeline`.
    """

    def get(self, graph: Graph, keys: List[type]) -> Any:
        """
        Compute the result for given keys from the graph.

        Must raise :py:class:`sciline.scheduler.CycleError` if the graph contains
        a cycle.
        """
        ...


class NaiveScheduler:
    """
    A naive scheduler that computes intermediate results and results in order.

    May consume excessive memory since intermediate results are not freed eagerly,
    but kept until returning the final result. Prefer installing `dask` and using
    :py:class:`DaskScheduler` instead.
    """

    def get(self, graph: Graph, keys: List[type]) -> Any:
        import graphlib

        dependencies = {tp: set(args.values()) for tp, (_, args) in graph.items()}
        ts = graphlib.TopologicalSorter(dependencies)
        try:
            # Create list from generator to force early exception if there is a cycle
            tasks = list(ts.static_order())
        except graphlib.CycleError as e:
            raise CycleError from e
        results: Dict[type, Any] = {}
        for t in tasks:
            provider, args = graph[t]
            args = {name: results[arg] for name, arg in args.items()}
            results[t] = provider(**args)
        return tuple(results[key] for key in keys)


class DaskScheduler:
    """Wrapper for a Dask scheduler.

    Note that this currently only works if all providers support posargs.
    """

    def __init__(self, scheduler: Optional[Callable[..., Any]] = None) -> None:
        """Wrap a dask scheduler or the default `dask.threaded.get`.

        Parameters
        ----------
        scheduler:
            A Dask scheduler, such as `dask.get`, `dask.threaded.get`,
            `dask.multiprocessing.get, or `dask.distributed.Client.get`.
        """
        if scheduler is None:
            import dask

            self._dask_get = dask.threaded.get
        else:
            self._dask_get = scheduler

    def get(self, graph: Graph, keys: List[type]) -> Any:
        dsk = {tp: (provider, *args.values()) for tp, (provider, args) in graph.items()}
        try:
            return self._dask_get(dsk, keys)
        except RuntimeError as e:
            if str(e).startswith("Cycle detected"):
                raise CycleError from e
            raise
