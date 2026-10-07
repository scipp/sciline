# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Stages cut a pipeline at a set of input keys, hold the part that does not
depend on them, and compute the part that does on each call."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx

from ._provider import Provider
from ._utils import key_name
from .data_graph import to_task_graph
from .handler import HandleAsComputeTimeException
from .pipeline import Pipeline
from .scheduler import Scheduler
from .task_graph import scheduler_or_default
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
    on first use or by :py:func:`build_stages`, and the values at the frontier are
    held. Each call to :py:meth:`compute` supplies values for the inputs and computes
    only what lies downstream of them. An input may be a parameter or an
    intermediate result; in both cases its own provider and ancestors are cut off.
    An output that is also an input is passed through.

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
        # The per-call part with the held values as parameters, built once when
        # warmed, since the held values do not change.
        self._call_graph: Graph = {}
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
        parameter that is not an input is in ``keys`` exactly when changing it on the
        pipeline would change the results of the stage.
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
        :py:func:`build_stages` to compute it together with that of other stages.
        """
        self._warm_up()
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
            groups={
                'Computed once, not kept': (
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
        self._warm_up()
        graph = dict(self._call_graph)
        for k, v in values.items():
            graph[k] = Provider.parameter(v)
        return _compute(graph, self._outputs, self._scheduler)

    def _warm_up(self) -> None:
        with self._lock:
            if not self._warm:
                self._hold(
                    _compute(self._static_graph, self._frontier, self._scheduler)
                )

    def _hold(self, values: Mapping[Key, Any]) -> None:
        """Hold the values at the frontier, taken from ``values``."""
        self._static = {k: values[k] for k in self._frontier}
        self._call_graph = {
            **self._dynamic_graph,
            **{k: Provider.parameter(v) for k, v in self._static.items()},
        }
        self._warm = True


@dataclass(frozen=True, kw_only=True)
class StageSpec:
    """Inputs and outputs of a stage, for building several stages with
    :py:func:`build_stages`."""

    inputs: tuple[Key, ...]
    """Keys supplied on each call."""
    outputs: tuple[Key, ...]
    """Keys whose values the stage computes."""


def build_stages(
    pipeline: Pipeline,
    specs: Iterable[StageSpec],
    *,
    scheduler: Scheduler | None = None,
) -> tuple[Stage, ...]:
    """Build several stages from one pipeline and compute their held parts in one run.

    Intermediate results shared by the held parts are computed once and released;
    each stage keeps only the values at its frontier, as a :py:class:`Stage` built
    on its own does. The stages are returned warm, so the first call to
    :py:meth:`Stage.compute` computes only the per-call part.

    Pass all stages of a driver, such as those of nested loops, in one call. Only
    then can stages that would hold a value the driver varies be detected.

    Parameters
    ----------
    pipeline:
        Pipeline with all parameters set that the outputs need, except the inputs.
    specs:
        Inputs and outputs of each stage.
    scheduler:
        Scheduler for computing the held parts and, in each stage, the per-call part.
        If not given, the default of :py:class:`Stage` is used.

    Returns
    -------
    :
        One stage per spec, in the order of ``specs``.

    Raises
    ------
    ValueError
        If a stage holds a value that depends on a parameter that another stage
        takes as input. The held value is for the one value of the parameter set on
        the pipeline, or for none, while the driver varies it. If the stage runs
        inside the loop over that input, it must take the value as an input,
        computed once per iteration of the loop. Otherwise the held value must not
        depend on that input, which means changing the pipeline. Also raised as by
        :py:class:`Stage` for an invalid spec.
    """
    scheduler = scheduler_or_default(scheduler)
    stages = tuple(
        Stage(pipeline, outputs=s.outputs, inputs=s.inputs, scheduler=scheduler)
        for s in specs
    )
    for i, held in enumerate(stages):
        params = {
            k
            for k, p in held._static_graph.items()
            if p.kind in ('parameter', 'unsatisfied')
        }
        for j, other in enumerate(stages):
            varied = params & set(other.inputs)
            if j != i and varied:
                raise ValueError(
                    f'specs[{i}] holds values that depend on '
                    f'{sorted(varied, key=str)}, which specs[{j}] takes as inputs. '
                    f'If specs[{i}] runs inside the loop over them, it must take '
                    'those values as inputs, computed once per iteration of that loop. '
                    f'Otherwise, change the pipeline so that the values specs[{i}] '
                    'holds do not depend on them'
                )
    # The stages are cut from one pipeline, so they agree on every key they share.
    graph: Graph = {}
    for stage in stages:
        graph.update(stage._static_graph)
    keys = tuple(dict.fromkeys(k for stage in stages for k in stage.frontier))
    values = _compute(graph, keys, scheduler)
    for stage in stages:
        stage._hold(values)
    return stages


def visualize_stages(
    *stages: Stage,
    groups: Mapping[str, tuple[Mapping[str, str], Iterable[Key]]] | None = None,
    show_legend: bool = True,
    show_held_ancestors: bool = True,
    **kwargs: Any,
) -> graphviz.Digraph:
    """Draw the keys that several stages use, such as the stages of one driver.

    By default, the nodes that each stage computes per call are filled with a color
    per stage, labeled in the legend by the inputs of the stage. The
    held part, the inputs, and the outputs are marked as by :py:meth:`Stage.visualize`;
    values one stage passes to another are outputs of the one and inputs of the other.
    Inputs that no stage computes are drawn as parameters.

    Parameters
    ----------
    stages:
        Stages cut from one pipeline, such as those returned by
        :py:func:`build_stages`.
    groups:
        Groups of nodes to style instead of the default, each by its label in the
        legend: a graphviz node style and the keys in the group. Where a key is in
        several groups, the styles are merged in the order of the groups. The styles
        used by default are in :py:mod:`sciline.visualize`, for example
        ``sciline.visualize.INPUT_STYLE``.
    show_legend:
        If True, add a legend with one entry per group that has a drawn node.
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
    if groups is None:
        groups = _groups_by_stage(stages)
    return _to_graphviz_with_parts(graph, groups, show_legend=show_legend, **kwargs)


def _groups_by_stage(
    stages: Sequence[Stage],
) -> dict[str, tuple[Mapping[str, str], Iterable[Key]]]:
    from .visualize import (
        FRONTIER_STYLE,
        HELD_STYLE,
        INPUT_STYLE,
        OUTPUT_STYLE,
        STAGE_FILLS,
    )

    computed = {k for stage in stages for k in stage._dynamic_graph}
    frontier = {k for stage in stages for k in stage.frontier}
    groups: dict[str, tuple[Mapping[str, str], Iterable[Key]]] = {
        'Computed once, not kept': (
            HELD_STYLE,
            {k for stage in stages for k in stage._static_graph} - frontier,
        ),
        'Held value': (FRONTIER_STYLE, frontier),
    }
    own_inputs: list[Key] = []
    for i, stage in enumerate(stages):
        own_inputs += [k for k in stage.inputs if k not in computed]
        names = ', '.join(key_name(k) for k in stage.inputs)
        label = f'Stage {i}, per call' + (f' with {names}' if names else '')
        fill = {'style': 'filled', 'fillcolor': STAGE_FILLS[i % len(STAGE_FILLS)]}
        groups[label] = (fill, stage._dynamic_graph)
    groups['Input'] = (INPUT_STYLE, own_inputs)
    groups['Output'] = (OUTPUT_STYLE, {k for stage in stages for k in stage.outputs})
    return groups
