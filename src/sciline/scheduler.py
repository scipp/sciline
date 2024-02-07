# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import inspect
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from sciline.typing import Graph, Key


class CycleError(Exception):
    pass


class Scheduler(Protocol):
    """
    Scheduler interface compatible with :py:class:`sciline.Pipeline`.
    """

    def get(self, graph: Graph, keys: List[Key]) -> Tuple[Any, ...]:
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

    def get(self, graph: Graph, keys: List[Key]) -> Tuple[Any, ...]:
        import graphlib

        dependencies = {
            tp: tuple(provider.arg_spec.keys()) for tp, provider in graph.items()
        }
        ts = graphlib.TopologicalSorter(dependencies)
        try:
            # Create list from generator to force early exception if there is a cycle
            tasks = list(ts.static_order())
        except graphlib.CycleError as e:
            raise CycleError from e
        results: Dict[Key, Any] = {}
        for t in tasks:
            results[t] = graph[t].call(results)
        return tuple(results[key] for key in keys)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


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

    def get(self, graph: Graph, keys: List[Key]) -> Any:
        from dask.utils import apply

        # Use `apply` to allow passing keyword arguments.
        # Contrary to the Dask docs, we need to pass positional args as a list
        # and keyword args with a nested tuple+list structure.
        # Otherwise, Dask would treat them as literal values instead of
        # references to other nodes.
        dsk = {
            tp: (
                apply,
                provider.func,
                list(provider.arg_spec.args),
                (dict, [[key, val] for key, val in provider.arg_spec.kwargs]),
            )
            for tp, provider in graph.items()
        }
        try:
            return self._dask_get(dsk, keys)
        except RuntimeError as e:
            if str(e).startswith("Cycle detected"):
                raise CycleError from e
            raise

    def __repr__(self) -> str:
        module = getattr(inspect.getmodule(self._dask_get), '__name__', '')
        name = self._dask_get.__name__
        return f'{self.__class__.__name__}({module}.{name})'
