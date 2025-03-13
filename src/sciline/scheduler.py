# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import inspect
from collections.abc import Callable, Hashable
from typing import Any, Protocol, runtime_checkable

from sciline.typing import Graph

from ._utils import key_full_qualname
from .reporter import NullReporter, Reporter


class CycleError(Exception):
    pass


@runtime_checkable
class Scheduler(Protocol):
    """
    Scheduler interface compatible with :py:class:`sciline.Pipeline`.
    """

    def get(
        self, graph: Graph, keys: list[Hashable], reporter: Reporter | None = None
    ) -> tuple[Any, ...]:
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

    def get(
        self, graph: Graph, keys: list[Hashable], reporter: Reporter | None = None
    ) -> tuple[Any, ...]:
        import graphlib

        reporter = _reporter_or_fallback(reporter)

        dependencies = {
            tp: tuple(provider.arg_spec.keys()) for tp, provider in graph.items()
        }
        ts = graphlib.TopologicalSorter(dependencies)
        try:
            # Create list from generator to force early exception if there is a cycle
            tasks = list(ts.static_order())
        except graphlib.CycleError as e:
            raise CycleError from e

        results: dict[Hashable, Any] = {}
        with reporter.run_computation(graph.values()):
            for t in tasks:
                results[t] = reporter.call_provider_with_reporting(graph[t], results)
        return tuple(results[key] for key in keys)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class DaskScheduler:
    """Wrapper for a Dask scheduler.

    Note that this currently only works if all providers support posargs.
    """

    def __init__(self, scheduler: Callable[..., Any] | None = None) -> None:
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

    def get(
        self, graph: Graph, keys: list[Hashable], reporter: Reporter | None = None
    ) -> Any:
        from dask.utils import apply

        reporter = _reporter_or_fallback(reporter)

        # Use `apply` to allow passing keyword arguments.
        # Contrary to the Dask docs, we need to pass positional args as a list
        # and keyword args with a nested tuple+list structure.
        # Otherwise, Dask would treat them as literal values instead of
        # references to other nodes.
        dsk = {
            _to_dask_key(tp): (
                apply,
                reporter.reporting_provider_func(provider),
                list(map(_to_dask_key, provider.arg_spec.args)),
                (
                    dict,
                    [[key, _to_dask_key(val)] for key, val in provider.arg_spec.kwargs],
                ),
            )
            for tp, provider in graph.items()
        }
        with reporter.run_computation(graph.values()):
            try:
                return self._dask_get(dsk, list(map(_to_dask_key, keys)))
            except RuntimeError as e:
                if str(e).startswith("Cycle detected"):
                    raise CycleError from e
                raise

    def __repr__(self) -> str:
        module = getattr(inspect.getmodule(self._dask_get), '__name__', '')
        name = self._dask_get.__name__
        return f'{self.__class__.__name__}({module}.{name})'


def _to_dask_key(key: Hashable) -> str:
    """Map a Sciline key to a dask key.

    According to the docs (https://docs.dask.org/en/stable/spec.html#definitions),
    keys in a Dask graph are not allowed to be types.
    So this function converts Sciline keys (types) to strings.
    """
    return key_full_qualname(key)


def _reporter_or_fallback(reporter: Reporter | None) -> Reporter:
    if reporter is None:
        return NullReporter()
    return reporter
