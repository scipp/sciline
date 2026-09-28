# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Stages cut a pipeline at a set of input keys, hold the part that does not
depend on them, and compute the part that does on each call."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import networkx as nx

from ._provider import Provider
from .data_graph import to_task_graph
from .handler import HandleAsComputeTimeException
from .pipeline import Pipeline
from .scheduler import Scheduler, scheduler_or_default
from .typing import Graph, Key

if TYPE_CHECKING:
    import graphviz


def _dependency_graph(graph: Graph) -> nx.DiGraph:
    g = nx.DiGraph()
    for key, provider in graph.items():
        g.add_node(key)
        for arg in provider.arg_spec.keys():
            g.add_edge(arg, key)
    return g


def _compute(graph: Graph, keys: Sequence[Key], scheduler: Scheduler) -> dict[Key, Any]:
    return dict(zip(keys, scheduler.get(graph, list(keys)), strict=True))


class Stage:
    """The part of a pipeline from a set of input keys to a set of output keys.

    Everything the outputs need that does not depend on the inputs is computed once,
    on first use, and the values at the frontier are held. Each call to
    :py:meth:`compute` supplies values for the inputs and computes only what lies
    downstream of them. An input may be a parameter or an intermediate result; in
    both cases its own provider and ancestors are cut off. An output that is also an
    input is passed through.

    A stage is a snapshot of the pipeline at the time it is built. Later changes to
    the pipeline do not affect it. Parameter values are held by reference, not
    copied, so modifying a value in place can change what the stage computes.

    :py:meth:`compute` may be called from several threads at once; the held part is
    computed once even then.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        outputs: Iterable[Key],
        inputs: Iterable[Key],
        scheduler: Scheduler | None = None,
    ) -> None:
        """
        Parameters
        ----------
        pipeline:
            Pipeline with all parameters set that the outputs need, except the inputs.
        outputs:
            Keys whose values the stage computes.
        inputs:
            Keys supplied on each call. Each must be needed by the outputs.
        scheduler:
            Scheduler for computing the held and the per-call parts. If not given,
            :py:class:`sciline.scheduler.DaskScheduler` is used if dask is installed,
            otherwise :py:class:`sciline.scheduler.NaiveScheduler`.

        Raises
        ------
        ValueError
            If an output is not in the pipeline or an input is not needed by the
            outputs.
        """
        self._outputs = tuple(outputs)
        self._inputs = tuple(inputs)
        self._scheduler = scheduler_or_default(scheduler)
        unknown = [k for k in self._outputs if k not in pipeline.underlying_graph]
        if unknown:
            raise ValueError(f'Outputs {unknown} are not in the pipeline')
        graph = to_task_graph(
            pipeline, targets=self._outputs, handler=HandleAsComputeTimeException()
        )
        deps = _dependency_graph(graph)
        # Cut off the providers of the inputs, then drop what only they needed.
        for key in self._inputs:
            if key in deps:
                deps.remove_edges_from(list(deps.in_edges(key)))
        needed_by_outputs = set(self._outputs)
        for key in self._outputs:
            needed_by_outputs |= nx.ancestors(deps, key)
        deps = deps.subgraph(needed_by_outputs)
        graph = {k: p for k, p in graph.items() if k in needed_by_outputs}
        missing = [k for k in self._inputs if k not in deps]
        if missing:
            raise ValueError(
                f'Inputs {missing} are not needed by outputs {self._outputs}'
            )
        dynamic: set[Key] = set(self._inputs)
        for key in self._inputs:
            dynamic |= nx.descendants(deps, key)
        self._dynamic_graph: Graph = {
            k: p for k, p in graph.items() if k in dynamic and k not in self._inputs
        }
        # The frontier: held nodes read by dynamic nodes, plus held outputs.
        frontier = dict.fromkeys(
            [
                arg
                for p in self._dynamic_graph.values()
                for arg in p.arg_spec.keys()
                if arg not in dynamic
            ]
            + [o for o in self._outputs if o not in dynamic]
        )
        self._frontier = tuple(frontier)
        self._dynamic = tuple(k for k in graph if k in dynamic)
        self._dynamic_outputs = tuple(o for o in self._outputs if o in dynamic)
        needed = set(self._frontier)
        for key in self._frontier:
            needed |= nx.ancestors(deps, key)
        self._static_graph = {k: p for k, p in graph.items() if k in needed}
        self._keys = frozenset(self._static_graph) | frozenset(self._dynamic)
        self._static: dict[Key, Any] = {}
        self._warm = False
        self._lock = threading.Lock()

    @property
    def inputs(self) -> tuple[Key, ...]:
        """Keys supplied on each call."""
        return self._inputs

    @property
    def outputs(self) -> tuple[Key, ...]:
        """Keys whose values the stage computes."""
        return self._outputs

    @property
    def keys(self) -> frozenset[Key]:
        """Keys the stage uses: those of the held part and of the per-call part.

        Ancestors that an intermediate input cuts off are not included, so a
        parameter is in ``keys`` exactly when changing it would change the stage.
        """
        return self._keys

    @property
    def frontier(self) -> tuple[Key, ...]:
        """Keys whose values are held: those read by the per-call part, and outputs
        that do not depend on the inputs."""
        return self._frontier

    @property
    def dynamic(self) -> tuple[Key, ...]:
        """Keys that depend on the inputs, including the inputs themselves."""
        return self._dynamic

    @property
    def dynamic_outputs(self) -> tuple[Key, ...]:
        """Outputs that depend on the inputs; the rest are fixed when the stage is
        built."""
        return self._dynamic_outputs

    def static(self) -> Mapping[Key, Any]:
        """Values at the frontier, computed on first use and held.

        The first call computes the held part, which may be expensive; use
        :py:func:`warm` to compute it together with that of other stages.
        """
        warm(self)
        return self._static

    def visualize(
        self,
        *,
        show_legend: bool = True,
        show_held_ancestors: bool = True,
        **kwargs: Any,
    ) -> graphviz.Digraph:
        """Draw the graph of the stage, with its inputs, held part, and per-call part.

        Parameters
        ----------
        show_legend:
            If True, add a legend explaining the node styles.
        show_held_ancestors:
            If False, draw the held values without what they were computed from.
        kwargs:
            Keyword arguments passed to :py:func:`sciline.visualize.to_graphviz`.
        """
        from .visualize import (
            DYNAMIC_STYLE,
            FRONTIER_STYLE,
            HELD_STYLE,
            INPUT_STYLE,
            OUTPUT_STYLE,
        )

        return visualize_stages(
            self,
            parts={
                'Held, computed once': (
                    HELD_STYLE,
                    set(self._static_graph) - set(self._frontier),
                ),
                'Held value': (FRONTIER_STYLE, self._frontier),
                'Input': (INPUT_STYLE, self._inputs),
                'Computed per call': (DYNAMIC_STYLE, self._dynamic_graph),
                'Output': (OUTPUT_STYLE, self._outputs),
            },
            show_legend=show_legend,
            show_held_ancestors=show_held_ancestors,
            **kwargs,
        )

    def compute(self, values: Mapping[Key, Any]) -> dict[Key, Any]:
        """Compute the values of the outputs from the values of the inputs.

        Parameters
        ----------
        values:
            A value for each input key, and nothing else.

        Returns
        -------
        :
            The value of each output key.

        Raises
        ------
        ValueError
            If the keys of ``values`` are not exactly the inputs.
        """
        if set(values) != set(self._inputs):
            raise ValueError(f'Expected values for {self._inputs}, got {tuple(values)}')
        graph: Graph = dict(self._dynamic_graph)
        for k, v in values.items():
            graph[k] = Provider.parameter(v)
        for k, v in self.static().items():
            graph[k] = Provider.parameter(v)
        return _compute(graph, self._outputs, self._scheduler)


def visualize_stages(
    *stages: Stage,
    parts: Mapping[str, tuple[Mapping[str, str], Iterable[Key]]],
    show_legend: bool = True,
    show_held_ancestors: bool = True,
    **kwargs: Any,
) -> graphviz.Digraph:
    """Draw the keys that several stages use, styling the nodes of each part.

    Inputs that no stage computes are drawn as parameters. The styles used by
    :py:meth:`Stage.visualize` and :py:meth:`Aggregation.visualize` are in
    :py:mod:`sciline.visualize`, for example ``sciline.visualize.INPUT_STYLE``.

    Parameters
    ----------
    stages:
        Stages that agree on every key they share, as for :py:func:`warm`.
    parts:
        For each part, by its label in the legend: a graphviz node style and the
        keys in the part. Where a key is in several parts, the styles are merged
        in the order of the parts.
    show_legend:
        If True, add a legend with one entry per part that has a drawn node.
    show_held_ancestors:
        If False, draw the held values without what they were computed from.
    kwargs:
        Keyword arguments passed to :py:func:`sciline.visualize.to_graphviz`.
    """
    from .visualize import _to_graphviz_with_parts

    graph: Graph = {}
    for stage in stages:
        graph.update(stage._static_graph)
        graph.update(stage._dynamic_graph)
    if not show_held_ancestors:
        frontier = {k for stage in stages for k in stage.frontier}
        dynamic = {k for stage in stages for k in stage._dynamic_graph}
        graph = {
            k: Provider.parameter(None) if k in frontier else p
            for k, p in graph.items()
            if k in frontier or k in dynamic
        }
    for stage in stages:
        for key in stage.inputs:
            graph.setdefault(key, Provider.parameter(None))
    return _to_graphviz_with_parts(graph, parts, show_legend=show_legend, **kwargs)


def _same_provider(a: Provider, b: Provider) -> bool:
    # Parameter and unsatisfied providers are rebuilt for each stage, so they are
    # compared by what they return. Parameter values are held by reference, so
    # identity tells whether they were set separately.
    if a.kind != b.kind:
        return False
    if a.kind == 'parameter':
        return a.func() is b.func()
    if a.kind == 'unsatisfied':
        return True
    return a.func is b.func and tuple(a.arg_spec.keys()) == tuple(b.arg_spec.keys())


def warm(*stages: Stage) -> None:
    """Compute the held parts of several stages in one run.

    Intermediate results shared by the held parts are computed once and released;
    each stage keeps only the values at its frontier, as when warmed on its own.
    Stages that are already warm are skipped.

    The stages may be built from different pipelines, such as copies of one
    pipeline with different parameter values, as long as they agree on every key
    they share. The scheduler of the first stage that is not yet warm is used.

    May be called from several threads at once, also together with
    :py:meth:`Stage.compute`; each held part is computed once.

    Parameters
    ----------
    stages:
        Stages that agree on every key they share.

    Raises
    ------
    ValueError
        If two stages compute a shared key differently, for example from different
        parameter values.
    """
    # Locks are taken in a fixed order, so that concurrent calls over overlapping
    # stages cannot deadlock.
    unique = tuple(dict.fromkeys(stages))
    with ExitStack() as locks:
        for stage in sorted(unique, key=id):
            locks.enter_context(stage._lock)
        cold = [s for s in unique if not s._warm]
        if not cold:
            return
        graph: Graph = {}
        keys: dict[Key, None] = {}
        for stage in cold:
            for key, provider in stage._static_graph.items():
                if key in graph and not _same_provider(graph[key], provider):
                    raise ValueError(
                        f'Stages compute {key} differently; warm them separately'
                    )
                graph[key] = provider
            keys.update(dict.fromkeys(stage._frontier))
        values = _compute(graph, tuple(keys), cold[0]._scheduler)
        for stage in cold:
            stage._static = {k: values[k] for k in stage._frontier}
            stage._warm = True
