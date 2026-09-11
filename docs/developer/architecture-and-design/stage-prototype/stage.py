"""Prototype: map/reduce outside the graph.

Built on a flat sciline pipeline, without cyclebane, map, or reduce:

Stage
    The part of a pipeline from a set of *input* keys to a set of *output* keys.
    Everything the outputs need that does not depend on the inputs is computed once,
    on first use, and held.  Calling the stage supplies values for the inputs and
    computes only what lies downstream of them.  An input may be a leaf (a parameter)
    or an intermediate node; in both cases its own providers and ancestors are cut
    off.  An output that is also an input is passed through.

Connectors
    Objects between stages with ``push``, ``value``, and ``clear``.  The type of the
    connector states why the graph is cut there: a reducer where members are
    combined, a ``Forwarder`` where a value is held between changes, such as the
    context of ``StreamProcessor``.  The accumulators of ``ess.reduce.streaming``
    are reducers in this sense.

Fold
    The table-fold shape over one flat pipeline: member keys, cut keys with an n-ary
    combine per key, outputs.  Two stages, ``contribute`` (member keys to cut) and
    ``finalize`` (cut to outputs), and the combine functions between them.  A fold
    holds nothing but its stages; parameters are set on the pipeline before the fold
    is built, and whoever loops over members owns the contributions.  ``compute``
    is that loop for a table.  The three entry points can run in different
    processes with the contribution serialized between them.

Nothing here adds nodes to the author's graph or hides a parameter: every object is
derived from the flat pipeline at the time it is built.  ``Stage`` and ``Fold`` are
mechanism and belong in sciline; the connectors are policy and belong next to
``StreamProcessor``.
Where several folds share a pipeline and a finalize, as sample and background runs
in esssans do, the package's own object holds the pipeline, the folds, the shared
finalize stage, and the contributions; see ``loki_validation.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any

import networkx as nx
import sciline
from sciline._provider import ArgSpec, Provider
from sciline.handler import HandleAsComputeTimeException
from sciline.scheduler import NaiveScheduler, Scheduler

Key = Hashable
Graph = dict[Key, Provider]


def _task_graph(pipeline: sciline.Pipeline, outputs: Sequence[Key]) -> Graph:
    # Prototype shortcut: the concrete task graph, with generics instantiated.
    # In v2 this is the graph the pipeline holds directly. Missing values (the
    # inputs, typically) are deferred to compute time, so the structure is complete.
    graph = pipeline.get(tuple(outputs), handler=HandleAsComputeTimeException())._graph
    return dict(graph)


def _dependency_graph(graph: Graph) -> nx.DiGraph:
    g = nx.DiGraph()
    for key, provider in graph.items():
        g.add_node(key)
        for arg in provider.arg_spec.keys():
            g.add_edge(arg, key)
    return g


class Stage:
    def __init__(
        self,
        pipeline: sciline.Pipeline,
        *,
        outputs: Iterable[Key],
        inputs: Iterable[Key],
        scheduler: Scheduler | None = None,
    ) -> None:
        self._outputs = tuple(outputs)
        self._inputs = tuple(inputs)
        self._scheduler = scheduler or NaiveScheduler()
        graph = _task_graph(pipeline, self._outputs)
        deps = _dependency_graph(graph)
        missing = [k for k in self._inputs if k not in deps]
        if missing:
            raise ValueError(f'Inputs {missing} are not needed by outputs {self._outputs}')
        dynamic: set[Key] = set(self._inputs)
        for key in self._inputs:
            dynamic |= nx.descendants(deps, key)
        self._dynamic_graph: Graph = {
            k: p for k, p in graph.items() if k in dynamic and k not in self._inputs
        }
        # The frontier: static nodes read by dynamic nodes, plus static outputs.
        static_needed = dict.fromkeys(
            [
                arg
                for k, p in self._dynamic_graph.items()
                for arg in p.arg_spec.keys()
                if arg not in dynamic
            ]
            + [o for o in self._outputs if o not in dynamic]
        )
        self.frontier = tuple(static_needed)
        self.dynamic = tuple(k for k in graph if k in dynamic)
        self.keys = frozenset(graph)
        needed = set(self.frontier)
        for key in self.frontier:
            needed |= nx.ancestors(deps, key)
        self._static_graph = {k: p for k, p in graph.items() if k in needed}
        self._static: dict[Key, Any] | None = None

    @property
    def static(self) -> dict[Key, Any]:
        """Values at the frontier, computed on first use and held."""
        if self._static is None:
            self._static = _compute(self._static_graph, self.frontier, self._scheduler)
        return self._static

    @property
    def inputs(self) -> tuple[Key, ...]:
        return self._inputs

    @property
    def outputs(self) -> tuple[Key, ...]:
        return self._outputs

    @property
    def dynamic_outputs(self) -> tuple[Key, ...]:
        """Outputs that depend on the inputs; the rest are fixed at build time."""
        dynamic = set(self.dynamic)
        return tuple(o for o in self._outputs if o in dynamic)

    def __call__(self, values: Mapping[Key, Any]) -> dict[Key, Any]:
        if set(values) != set(self._inputs):
            raise ValueError(f'Expected values for {self._inputs}, got {tuple(values)}')
        graph: Graph = dict(self._dynamic_graph)
        for k, v in values.items():
            graph[k] = Provider.parameter(v)
        for k, v in self.static.items():
            graph[k] = Provider.parameter(v)
        return _compute(graph, self._outputs, self._scheduler)


def warm(*stages: Stage) -> None:
    """Compute the static parts of several stages of one pipeline in one run.

    Intermediates the static parts share are computed once and released; each
    stage keeps only its frontier, as it does when warmed on its own.
    """
    stages = [s for s in stages if s._static is None]
    if not stages:
        return
    graph: Graph = {}
    keys: dict[Key, None] = {}
    for stage in stages:
        graph.update(stage._static_graph)
        keys.update(dict.fromkeys(stage.frontier))
    values = _compute(graph, tuple(keys), stages[0]._scheduler)
    for stage in stages:
        stage._static = {k: values[k] for k in stage.frontier}


def _compute(graph: Graph, keys: Sequence[Key], scheduler: Scheduler) -> dict[Key, Any]:
    # One sink over all requested keys, so that a key consumed by another requested
    # key is not released by the scheduler before it is returned.
    sink = object()
    graph = dict(graph)
    graph[sink] = Provider(
        func=lambda *args: args, arg_spec=ArgSpec.from_args(*keys), kind='function'
    )
    (values,) = scheduler.get(graph, [sink])
    return dict(zip(keys, values, strict=True))


# --- Connectors -----------------------------------------------------------------


class Forwarder:
    """Holds the latest value pushed; the connector for a context between stages."""

    def __init__(self) -> None:
        self._value: Any = None
        self._set = False

    def push(self, value: Any) -> None:
        self._value = value
        self._set = True

    @property
    def value(self) -> Any:
        if not self._set:
            raise ValueError('Nothing has been pushed')
        return self._value

    def clear(self) -> None:
        self._value = None
        self._set = False


# --- Fold -----------------------------------------------------------------------

Combine = Callable[..., Any]
Contribution = dict[Key, Any]
"""Values at the cut keys, per member or combined."""


class Fold:
    def __init__(
        self,
        pipeline: sciline.Pipeline,
        *,
        members: Iterable[Key],
        at: Mapping[Key, Combine],
        outputs: Iterable[Key] = (),
        scheduler: Scheduler | None = None,
    ) -> None:
        """
        Parameters
        ----------
        pipeline:
            The flat pipeline with its parameters set; the fold is a snapshot of it.
        members:
            The keys supplied per member, the columns of a member table.
        at:
            The cut keys with their n-ary combine function.  A cut key that does not
            depend on the members is not folded; finalize computes it from the fixed
            part of the graph.
        outputs:
            Keys computed by finalize from the cut.  Omit for a fold used only for
            its contributions, such as one of several sharing a finalize stage.
        """
        members = tuple(members)
        outputs = tuple(outputs)
        probe = Stage(pipeline, outputs=tuple(at), inputs=members)
        self.cut = probe.dynamic_outputs
        self._at = {k: at[k] for k in self.cut}
        self.contribute_stage = Stage(
            pipeline, outputs=self.cut, inputs=members, scheduler=scheduler
        )
        self.finalize_stage = (
            Stage(pipeline, outputs=outputs, inputs=self.cut, scheduler=scheduler)
            if outputs
            else None
        )

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The stages to warm together."""
        if self.finalize_stage is None:
            return (self.contribute_stage,)
        return (self.contribute_stage, self.finalize_stage)

    def contribute(self, row: Mapping[Key, Any]) -> Contribution:
        return self.contribute_stage(row)

    def combine(self, contributions: Sequence[Contribution]) -> Contribution:
        return {k: f(*(c[k] for c in contributions)) for k, f in self._at.items()}

    def finalize(self, contribution: Contribution) -> dict[Key, Any]:
        if self.finalize_stage is None:
            raise ValueError('This fold has no outputs')
        return self.finalize_stage(contribution)

    def compute(self, table: Any) -> dict[Key, Any]:
        """Contribute per row, combine, finalize; nothing is held."""
        warm(*self.stages)
        rows = _rows(table)
        return self.finalize(self.combine([self.contribute(row) for row in rows.values()]))


def compute_members(
    pipeline: sciline.Pipeline, *, members: Iterable[Key], key: Key, table: Any
) -> dict[Hashable, Any]:
    """Per-member value of a key that depends on the member keys."""
    stage = Stage(pipeline, outputs=(key,), inputs=tuple(members))
    return {label: stage(row)[key] for label, row in _rows(table).items()}


def _rows(table: Any) -> dict[Hashable, dict[Key, Any]]:
    """A dict of columns, labeled by position, or a DataFrame, labeled by its index."""
    if hasattr(table, 'iterrows'):
        return {idx: dict(row.items()) for idx, row in table.iterrows()}
    keys = list(table)
    n = len(table[keys[0]])
    return {i: {k: table[k][i] for k in keys} for i in range(n)}
