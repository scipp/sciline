# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Stages cut a pipeline at a set of input keys, compute the part that does not
depend on them once, and compute the part that does on each call."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
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
    on first use or by :py:func:`build_stages`. Of this part, the stage holds the
    values that the per-call part reads, and the outputs that do not depend on the
    inputs. Each call to :py:meth:`compute` supplies values for the inputs and
    computes only what lies downstream of them. An input may be a parameter or an
    intermediate result; in both cases its own provider and ancestors are cut off.
    An output that is also an input is passed through.

    A stage is a snapshot of the pipeline at the time it is built. Later changes to
    the pipeline do not affect it. Parameter values are held by reference, not
    copied, so modifying a value in place can change what the stage computes.

    :py:meth:`compute` may be called from several threads at once. Even then, the
    part that does not depend on the inputs is computed only once.
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
            Scheduler for computing the held values and the per-call part. If not
            given, :py:class:`sciline.scheduler.DaskScheduler` is used if dask is
            installed, otherwise :py:class:`sciline.scheduler.NaiveScheduler`.

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
        per_call: set[Key] = set(self._inputs)
        for key in self._inputs:
            per_call |= nx.descendants(deps, key)
        self._per_call_graph: Graph = {
            k: p for k, p in graph.items() if k in per_call and k not in self._inputs
        }
        # The held keys: keys computed once that the per-call part reads, plus
        # outputs computed once.
        held = dict.fromkeys(
            [
                arg
                for p in self._per_call_graph.values()
                for arg in p.arg_spec.keys()
                if arg not in per_call
            ]
            + [o for o in self._outputs if o not in per_call]
        )
        self._held = tuple(held)
        self._per_call = tuple(k for k in graph if k in per_call)
        self._per_call_outputs = tuple(o for o in self._outputs if o in per_call)
        needed = set(self._held)
        for key in self._held:
            needed |= nx.ancestors(deps, key)
        self._once_graph = {k: p for k, p in graph.items() if k in needed}
        self._keys = frozenset(self._once_graph) | frozenset(self._per_call)
        self._held_values: dict[Key, Any] = {}
        # The per-call part with the held values as parameters, built once when
        # warmed, since the held values do not change.
        self._graph_with_held_values: Graph = {}
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
        """Keys the stage uses: those of the part computed once and of the per-call
        part.

        Ancestors that an intermediate input cuts off are not included, so a
        parameter that is not an input is in ``keys`` exactly when changing it on the
        pipeline would change the results of the stage.
        """
        return self._keys

    @property
    def held(self) -> tuple[Key, ...]:
        """Keys whose values are held: those read by the per-call part, and outputs
        that do not depend on the inputs."""
        return self._held

    @property
    def per_call(self) -> tuple[Key, ...]:
        """Keys whose values change with each call: the inputs and the keys that
        depend on them."""
        return self._per_call

    @property
    def per_call_outputs(self) -> tuple[Key, ...]:
        """Outputs that depend on the inputs; the rest are held."""
        return self._per_call_outputs

    def held_values(self) -> Mapping[Key, Any]:
        """Values of the held keys, computed on first use.

        The first call computes them, which may be expensive; use
        :py:func:`build_stages` to compute them together with those of other stages.
        """
        self._warm_up()
        return self._held_values

    def visualize(
        self,
        *,
        show_legend: bool = True,
        show_held_ancestors: bool = True,
        **kwargs: Any,
    ) -> graphviz.Digraph:
        """Draw the graph of the stage, with its inputs, the part computed once, the
        held values, and the per-call part.

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
            HELD_STYLE,
            INPUT_STYLE,
            ONCE_STYLE,
            OUTPUT_STYLE,
            PER_CALL_STYLE,
        )

        return visualize_stages(
            self,
            groups={
                'Computed once, not kept': (
                    ONCE_STYLE,
                    set(self._once_graph) - set(self._held),
                ),
                'Held value': (HELD_STYLE, self._held),
                'Input': (INPUT_STYLE, self._inputs),
                'Computed per call': (PER_CALL_STYLE, self._per_call_graph),
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
        graph = dict(self._graph_with_held_values)
        for k, v in values.items():
            graph[k] = Provider.parameter(v)
        return _compute(graph, self._outputs, self._scheduler)

    def _warm_up(self) -> None:
        with self._lock:
            if not self._warm:
                self._hold(_compute(self._once_graph, self._held, self._scheduler))

    def _hold(self, values: Mapping[Key, Any]) -> None:
        """Hold the values of the held keys, taken from ``values``."""
        self._held_values = {k: values[k] for k in self._held}
        self._graph_with_held_values = {
            **self._per_call_graph,
            **{k: Provider.parameter(v) for k, v in self._held_values.items()},
        }
        self._warm = True


def build_stages(
    pipeline: Pipeline,
    *,
    inputs: Iterable[Key],
    outputs: Iterable[Iterable[Key]],
    scheduler: Scheduler | None = None,
) -> tuple[Stage, ...]:
    """Build several stages from one pipeline and compute their held values in one
    run.

    Every stage is cut at every key in ``inputs``, except at its own outputs. Its
    inputs are those of ``inputs`` that its outputs still need after this cut. For
    example, with ``inputs=(Filename, Numerator)`` and
    ``outputs=[(Numerator,), (Result,)]``, the first stage computes ``Numerator``
    from ``Filename``, and the second computes ``Result`` from ``Numerator``.

    Since every stage is cut at every key in ``inputs``, no stage holds a value that
    depends on one of them. A stage that needs such a value, and does not get it
    through another key in ``inputs``, takes the key the value depends on as input.
    For example, a stage inside a loop over runs that reads a value depending on the
    run, which the outer stage does not pass on, takes ``Filename`` as input as
    well. A driver that does not supply it gets an error on the first call.

    Intermediate results shared by several stages are computed once and released;
    each stage keeps only its held values, as a :py:class:`Stage` built on
    its own does. The stages are returned warm, so the first call to
    :py:meth:`Stage.compute` computes only the per-call part.

    Parameters
    ----------
    pipeline:
        Pipeline with all parameters set that the outputs need, except the inputs.
    inputs:
        Keys at which the pipeline is cut, such as the keys a driver loops over and
        the values it passes from one stage to another.
    outputs:
        Keys whose values each stage computes, one tuple per stage.
    scheduler:
        Scheduler for computing the held values and, in each stage, the per-call
        part. If not given, the default of :py:class:`Stage` is used.

    Returns
    -------
    :
        One stage per tuple in ``outputs``, in the same order.

    Raises
    ------
    ValueError
        If a key in ``inputs`` is not an input of any stage, or an output is not in
        the pipeline.
    """
    inputs = tuple(inputs)
    outputs = [tuple(outs) for outs in outputs]
    targets = tuple(dict.fromkeys(k for outs in outputs for k in outs))
    unknown = [k for k in targets if k not in pipeline.underlying_graph]
    if unknown:
        raise ValueError(f'Outputs {unknown} are not in the pipeline')
    deps = _dependency_graph(
        to_task_graph(pipeline, targets=targets, handler=HandleAsComputeTimeException())
    )
    stage_inputs = []
    for outs in outputs:
        cut = deps.copy()
        for key in inputs:
            if key in cut and key not in outs:
                cut.remove_edges_from(list(cut.in_edges(key)))
        needed = set().union(*(nx.ancestors(cut, key) for key in outs))
        stage_inputs.append(tuple(k for k in inputs if k in needed and k not in outs))
    unused = [k for k in inputs if not any(k in ins for ins in stage_inputs)]
    if unused:
        raise ValueError(f'Inputs {unused} are not inputs of any stage')
    scheduler = scheduler_or_default(scheduler)
    stages = tuple(
        Stage(pipeline, outputs=outs, inputs=ins, scheduler=scheduler)
        for outs, ins in zip(outputs, stage_inputs, strict=True)
    )
    # The stages are cut from one pipeline, so they agree on every key they share.
    graph: Graph = {}
    for stage in stages:
        graph.update(stage._once_graph)
    keys = tuple(dict.fromkeys(k for stage in stages for k in stage.held))
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
    per stage, labeled in the legend by the inputs of the stage. The part computed
    once, the held values, the inputs, and the outputs are marked as by
    :py:meth:`Stage.visualize`; values one stage passes to another are outputs of the
    one and inputs of the other. Inputs that no stage computes are drawn as
    parameters.

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
        graph.update(stage._once_graph)
        graph.update(stage._per_call_graph)
    if not show_held_ancestors:
        held = {k for stage in stages for k in stage.held}
        per_call = {k for stage in stages for k in stage._per_call_graph}
        graph = {
            k: Provider.parameter(None) if k in held else p
            for k, p in graph.items()
            if k in held or k in per_call
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
        HELD_STYLE,
        INPUT_STYLE,
        ONCE_STYLE,
        OUTPUT_STYLE,
        STAGE_FILLS,
    )

    computed = {k for stage in stages for k in stage._per_call_graph}
    held = {k for stage in stages for k in stage.held}
    groups: dict[str, tuple[Mapping[str, str], Iterable[Key]]] = {
        'Computed once, not kept': (
            ONCE_STYLE,
            {k for stage in stages for k in stage._once_graph} - held,
        ),
        'Held value': (HELD_STYLE, held),
    }
    own_inputs: list[Key] = []
    for i, stage in enumerate(stages):
        own_inputs += [k for k in stage.inputs if k not in computed]
        names = ', '.join(key_name(k) for k in stage.inputs)
        label = f'Stage {i}, per call' + (f' with {names}' if names else '')
        fill = {'style': 'filled', 'fillcolor': STAGE_FILLS[i % len(STAGE_FILLS)]}
        groups[label] = (fill, stage._per_call_graph)
    groups['Input'] = (INPUT_STYLE, own_inputs)
    groups['Output'] = (OUTPUT_STYLE, {k for stage in stages for k in stage.outputs})
    return groups
