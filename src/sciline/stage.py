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
            groups={
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
            A value for each input key. Values for keys the stage does not use, those
            not in :py:attr:`keys`, are ignored, so a driver can pass on everything
            the stages of enclosing loops returned.

        Returns
        -------
        :
            The value of each output key.

        Raises
        ------
        ValueError
            If a value for an input is missing, or if a value is given for a key that
            the stage uses but does not take as input. The stage holds or computes
            such a key itself and would ignore the value.
        """
        missing = [k for k in self._inputs if k not in values]
        if missing:
            raise ValueError(f'Missing values for inputs {missing}')
        not_inputs = [k for k in values if k in self._keys and k not in self._inputs]
        if not_inputs:
            raise ValueError(
                f'The stage uses {not_inputs} but does not take them as inputs'
            )
        graph: Graph = dict(self._dynamic_graph)
        for k in self._inputs:
            graph[k] = Provider.parameter(values[k])
        for k, v in self.static().items():
            graph[k] = Provider.parameter(v)
        return _compute(graph, self._outputs, self._scheduler)


def visualize_stages(
    *stages: Stage,
    groups: Mapping[str, tuple[Mapping[str, str], Iterable[Key]]] | None = None,
    show_legend: bool = True,
    show_held_ancestors: bool = True,
    **kwargs: Any,
) -> graphviz.Digraph:
    """Draw the keys that several stages use, such as the stages of :py:func:`enclose`.

    By default, the nodes that each stage computes per call are filled with a color
    per stage, labeled in the legend by the inputs of the stage. The
    held part, the inputs, and the outputs are marked as by :py:meth:`Stage.visualize`;
    values one stage passes to another are outputs of the one and inputs of the other.
    Inputs that no stage computes are drawn as parameters.

    Parameters
    ----------
    stages:
        Stages that agree on every key they share, as for :py:func:`warm`.
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
        'Held, computed once': (
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


def enclose(
    pipeline: Pipeline,
    stages: Iterable[Stage],
    *,
    inputs: Iterable[Key],
    outputs: Iterable[Key] = (),
    scheduler: Scheduler | None = None,
) -> tuple[Stage, ...]:
    """Put stages inside a loop over ``inputs``.

    A stage holds the values at its frontier. Some of them may depend on ``inputs``,
    such as the content of a file, held by a stage that loops over the banks of the
    file. The returned outer stage computes these values from ``inputs``, once per
    iteration of the new loop. The returned inner stages take them as inputs, so the
    driver passes what the outer stage returned on to them. Values that do not depend
    on ``inputs`` stay held by the stages that read them.

    Nested loops are built from the inside out: build the stages of the innermost
    loop, enclose them, then enclose the result. Pass every stage inside the new loop
    each time, not only those of the next inner loop, since any of them may read a
    value that depends on ``inputs``. Enclose all stages of a loop in one call: a
    stage returned by ``enclose`` takes the forwarded values as inputs, and enclosing
    it again in a loop over the same inputs does not forward them.

    Parameters
    ----------
    pipeline:
        The pipeline the stages were built from.
    stages:
        The stages inside the new loop.
    inputs:
        Keys whose values the driver supplies on each iteration of the new loop.
    outputs:
        Keys the driver needs on each iteration of the new loop, such as accumulation
        keys, in addition to what the inner stages read.
    scheduler:
        Scheduler for the outer stage. The inner stages keep their schedulers.

    Returns
    -------
    :
        The outer stage, then the given stages rebuilt, in the given order.

    Raises
    ------
    ValueError
        If no stage reads a value that depends on ``inputs`` and ``outputs`` is empty,
        or if an output of a stage depends on ``inputs`` but not on the inputs of that
        stage. The driver would push such an output once per iteration of a loop that
        it does not vary in.

    Examples
    --------
    Sum ``Total`` over the detector banks of several files, reading each file once:

    .. code-block:: python

        bank_stage = Stage(pipeline, outputs=(Total,), inputs=(Bank,))
        file_stage, bank_stage = enclose(pipeline, [bank_stage], inputs=(Filename,))

        total = Reduced(operator.add)()
        for filename in filenames:
            held = file_stage.compute({Filename: filename})
            for bank in banks:
                total.push(bank_stage.compute({**held, Bank: bank})[Total])
    """
    stages = tuple(stages)
    inputs = tuple(inputs)
    outputs = tuple(outputs)
    frontier = tuple(dict.fromkeys(k for s in stages for k in s.frontier))
    graph = to_task_graph(
        pipeline, targets=frontier, handler=HandleAsComputeTimeException()
    )
    deps = _dependency_graph(graph)
    varying = [k for k in frontier if _upstream(deps, k) & set(inputs)]
    if not varying and not outputs:
        raise ValueError(f'No stage reads a value that depends on {inputs}')
    for i, stage in enumerate(stages):
        constant = [k for k in stage.outputs if k in varying]
        if constant:
            raise ValueError(
                f'Outputs {constant} of stages[{i}] depend on {inputs} but not on '
                f'the inputs of that stage. Remove them from its outputs and pass '
                'them as outputs of the enclosing stage'
            )
    outer = Stage(
        pipeline,
        outputs=(*outputs, *(k for k in varying if k not in outputs)),
        inputs=inputs,
        scheduler=scheduler,
    )
    inner = tuple(
        Stage(
            pipeline,
            outputs=s.outputs,
            inputs=(*(k for k in varying if k in s.frontier), *s.inputs),
            scheduler=s._scheduler,
        )
        for s in stages
    )
    # Stages are snapshots, so a pipeline changed since the given stages were built
    # would give stages that disagree with them.
    built = {k: p for s in stages for k, p in _graph(s).items()}
    for stage in (outer, *inner):
        for key, provider in _graph(stage).items():
            if key in built and not _same_provider(built[key], provider):
                raise ValueError(
                    f'The pipeline computes {key} differently than when the stages '
                    'were built'
                )
    return (outer, *inner)


def _upstream(deps: nx.DiGraph, key: Key) -> set[Key]:
    if key not in deps:
        return {key}
    return {key, *nx.ancestors(deps, key)}


def _graph(stage: Stage) -> Graph:
    return {**stage._static_graph, **stage._dynamic_graph}
