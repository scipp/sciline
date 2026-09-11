# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Stages cut a pipeline at a set of input keys, hold the part that does not
depend on them, and compute the part that does on each call."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import networkx as nx

from ._provider import Provider
from .data_graph import to_task_graph
from .handler import HandleAsComputeTimeException
from .pipeline import Pipeline
from .scheduler import Scheduler, scheduler_or_default
from .typing import Graph, Key


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
    on first use, and held. Calling the stage supplies values for the inputs and
    computes only what lies downstream of them. An input may be a parameter or an
    intermediate result; in both cases its own provider and ancestors are cut off.
    An output that is also an input is passed through.

    A stage is a snapshot of the pipeline at the time it is built. Later changes to
    the pipeline do not affect it.
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
            Keys computed by the stage.
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
        self._keys = frozenset(graph)
        needed = set(self._frontier)
        for key in self._frontier:
            needed |= nx.ancestors(deps, key)
        self._static_graph = {k: p for k, p in graph.items() if k in needed}
        self._static: dict[Key, Any] | None = None

    @property
    def inputs(self) -> tuple[Key, ...]:
        """Keys supplied on each call."""
        return self._inputs

    @property
    def outputs(self) -> tuple[Key, ...]:
        """Keys computed by the stage."""
        return self._outputs

    @property
    def keys(self) -> frozenset[Key]:
        """All keys the outputs depend on, including the outputs themselves."""
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

    @property
    def static(self) -> Mapping[Key, Any]:
        """Values at the frontier, computed on first use and held."""
        if self._static is None:
            self._static = _compute(self._static_graph, self._frontier, self._scheduler)
        return self._static

    def __call__(self, values: Mapping[Key, Any]) -> dict[Key, Any]:
        """Compute the outputs for the given input values.

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
        for k, v in self.static.items():
            graph[k] = Provider.parameter(v)
        return _compute(graph, self._outputs, self._scheduler)


def warm(*stages: Stage) -> None:
    """Compute the held parts of several stages of one pipeline in one run.

    Intermediate results shared by the held parts are computed once and released;
    each stage keeps only the values at its frontier, as when warmed on its own.
    Stages that are already warm are skipped.

    All stages must be built from the same pipeline. The scheduler of the first
    stage is used.

    Parameters
    ----------
    stages:
        Stages built from one pipeline.
    """
    cold = [s for s in stages if s._static is None]
    if not cold:
        return
    graph: Graph = {}
    keys: dict[Key, None] = {}
    for stage in cold:
        graph.update(stage._static_graph)
        keys.update(dict.fromkeys(stage._frontier))
    values = _compute(graph, tuple(keys), cold[0]._scheduler)
    for stage in cold:
        stage._static = {k: values[k] for k in stage._frontier}
