# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Stages cut a pipeline at a set of input keys, hold the part that does not
depend on them, and compute the part that does on each call."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import networkx as nx

from ._provider import Provider
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
    :py:meth:`Stage.visualize` are in :py:mod:`sciline.visualize`, for example
    ``sciline.visualize.INPUT_STYLE``.

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


@dataclass(frozen=True, eq=False)
class Part:
    """One level of a driver's loop, for :py:func:`split`.

    The driver supplies the values of ``inputs`` on each iteration of the part's loop,
    or on each update of a stream. ``parent`` is the part of the enclosing loop, whose
    values for the current iteration the driver also holds. Parts compare by
    identity.
    """

    inputs: tuple[Key, ...]
    """Keys whose values the driver supplies on each iteration."""
    outputs: tuple[Key, ...] = ()
    """Keys the driver needs on each iteration, such as accumulation keys. Each must
    depend on ``inputs``."""
    parent: Part | None = field(default=None, repr=False)
    """The part of the enclosing loop, if any."""

    def _path(self) -> tuple[Part, ...]:
        return (*(self.parent._path() if self.parent else ()), self)


def split(
    pipeline: Pipeline, *parts: Part, scheduler: Scheduler | None = None
) -> tuple[Stage, ...]:
    """Build a stage per part of nested loops, holding per-iteration work per level.

    A part reads values that do not depend on its inputs. Each such value is
    computed by the deepest ancestor whose inputs it depends on, once per iteration
    of that ancestor's loop, and passed to the part as an input. A value that
    depends on no part's inputs is held by the stage that reads it, as by any
    :py:class:`Stage`. Dependencies are taken with the graph cut at the inputs of
    the part and its ancestors: a part whose inputs are accumulation keys, such as
    one that runs after combining the members of an inner loop, does not depend on
    the inner loop's inputs.

    The stage of a part takes the values it reads from its ancestors and the part's
    inputs. Its outputs are those of the part, then the values its descendants read
    from it. The driver pushes only the outputs of the part.

    Parameters
    ----------
    pipeline:
        Pipeline with all parameters set that the outputs need, except the inputs of
        the parts.
    parts:
        The parts, including the ancestors of each.
    scheduler:
        Scheduler for all stages. If not given,
        :py:class:`sciline.scheduler.DaskScheduler` is used if dask is installed,
        otherwise :py:class:`sciline.scheduler.NaiveScheduler`.

    Returns
    -------
    :
        A stage per part, in the order of ``parts``.

    Raises
    ------
    ValueError
        If a part is given twice, if the ancestor of a part is not given, if an
        output is not in the pipeline, if a part has no outputs and no
        descendant reads from it, if an output of a part does not depend on the
        part's inputs (it would be pushed once per iteration of a loop that it does
        not vary in), or if a part reads a value that depends on the inputs of a part
        that is not its ancestor.
    """
    if len(set(parts)) != len(parts):
        raise ValueError('Each part must be given once')
    for part in parts:
        if any(p not in parts for p in part._path()):
            raise ValueError(f'An ancestor of {part} is not among the parts')
    targets = tuple(dict.fromkeys(k for p in parts for k in p.outputs))
    unknown = [k for k in targets if k not in pipeline.underlying_graph]
    if unknown:
        raise ValueError(f'Outputs {unknown} are not in the pipeline')
    full = _dependency_graph(
        to_task_graph(pipeline, targets=targets, handler=HandleAsComputeTimeException())
    )
    all_inputs = {k for p in parts for k in p.inputs}
    from_ancestors: dict[Part, list[Key]] = {p: [] for p in parts}
    for_descendants: dict[Part, list[Key]] = {p: [] for p in parts}
    # Descendants first, so that the outputs of a part are known when it is reached.
    for part in sorted(parts, key=lambda p: len(p._path()), reverse=True):
        path = part._path()
        path_inputs = {k for p in path for k in p.inputs}
        deps = _cut(full, path_inputs)
        constant = [k for k in part.outputs if not _upstream(deps, k) & set(part.inputs)]
        if constant:
            owners = ', '.join(
                f'{k} on {_owner(deps, k, path[:-1]) or "none of its ancestors"}'
                for k in constant
            )
            raise ValueError(
                f'Outputs {constant} of {part} do not depend on its inputs. '
                f'Declare each on the part whose inputs it depends on: {owners}'
            )
        outputs = (*part.outputs, *for_descendants[part])
        if not outputs:
            raise ValueError(f'{part} has no outputs and no part reads from it')
        probe = Stage(
            pipeline, outputs=outputs, inputs=part.inputs, scheduler=scheduler
        )
        for key in probe.frontier:
            foreign = (all_inputs - path_inputs) & _upstream(deps, key)
            if foreign:
                raise ValueError(
                    f'{part} reads {key}, which depends on {sorted(foreign, key=str)}, '
                    'the inputs of a part that is not its ancestor'
                )
            owner = _owner(deps, key, path[:-1])
            if owner is not None:
                from_ancestors[part].append(key)
                if key not in (*owner.outputs, *for_descendants[owner]):
                    for_descendants[owner].append(key)
    return tuple(
        Stage(
            pipeline,
            outputs=(*p.outputs, *for_descendants[p]),
            inputs=(*from_ancestors[p], *p.inputs),
            scheduler=scheduler,
        )
        for p in parts
    )


def _cut(graph: nx.DiGraph, keys: Iterable[Key]) -> nx.DiGraph:
    cut = graph.copy()
    for key in keys:
        if key in cut:
            cut.remove_edges_from(list(cut.in_edges(key)))
    return cut


def _upstream(deps: nx.DiGraph, key: Key) -> set[Key]:
    if key not in deps:
        return {key}
    return {key, *nx.ancestors(deps, key)}


def _owner(deps: nx.DiGraph, key: Key, ancestors: Sequence[Part]) -> Part | None:
    """The deepest of the ancestors whose inputs the key depends on."""
    upstream = _upstream(deps, key)
    return next((p for p in reversed(ancestors) if upstream & set(p.inputs)), None)
