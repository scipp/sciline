"""Prototype: map/reduce outside the graph.

Built on a flat sciline pipeline, without cyclebane, map, or reduce:

Stage
    The part of a pipeline from a set of *input* keys to a set of *output* keys.
    Everything the outputs need that does not depend on the inputs is computed once,
    on first use, and held.  Calling the stage supplies values for the inputs and
    computes only what lies downstream of them.  An input may be a leaf (a parameter)
    or an intermediate node; in both cases its own providers and ancestors are cut
    off.  An output that is also an input is passed through.

Accumulator
    The protocol for what sits between stages: ``push`` a value, read the combined
    ``value``.  The accumulators of ``ess.reduce.streaming`` satisfy it.  ``Buffered``
    adapts an n-ary function: it holds what was pushed and applies the function on
    ``value``.  Whether a combine buffers or runs incrementally is the accumulator's
    choice, not the aggregation's.

Aggregation
    The table-fold shape over one flat pipeline: member keys, accumulation keys with
    an accumulator per key, outputs.  Two stages, ``contribute`` (member keys to
    accumulation keys) and ``finalize`` (accumulation keys to outputs), with the
    accumulators between them.  An aggregation holds nothing but its stages;
    parameters are set on the pipeline before it is built, and whoever loops over
    members owns the contributions.  ``compute`` is that loop for a table, pushing
    each contribution as it is made.  The three entry points can run in different
    processes with the contribution serialized between them.

Nothing here adds nodes to the author's graph or hides a parameter: every object is
derived from the flat pipeline at the time it is built.  ``Stage``, ``Accumulator``,
``Buffered``, and ``Aggregation`` belong in sciline.  ``Forwarder``, which holds the
latest value pushed, is the context connector of ``StreamProcessor`` and belongs in
ess.reduce.  Where several aggregations share a pipeline and a finalize, as sample
and background runs in esssans do, the package's own object holds the pipeline, the
aggregations, the shared finalize stage, and the contributions; see
``loki_validation.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

import networkx as nx

import sciline
from sciline._provider import Provider
from sciline.handler import HandleAsComputeTimeException
from sciline.scheduler import NaiveScheduler, Scheduler

Key = Hashable
Graph = dict[Key, Provider]
T = TypeVar('T')


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
            raise ValueError(
                f'Inputs {missing} are not needed by outputs {self._outputs}'
            )
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
    return dict(zip(keys, scheduler.get(graph, list(keys)), strict=True))


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


# --- Accumulators ---------------------------------------------------------------


class Accumulator(Protocol[T]):
    """What sits between stages: values are pushed, the combined value is read."""

    def push(self, value: T) -> None: ...

    @property
    def value(self) -> T: ...


class Buffered(Generic[T]):
    """Factory for an accumulator that applies an n-ary function to all pushed values.

    Holds every pushed value until ``value`` is read: one pass for the function, the
    whole input in memory.  Right for concatenation, where no incremental form is
    cheaper; wrong for a large dense sum, which wants a running total.
    """

    def __init__(self, func: Callable[..., T]) -> None:
        self._func = func

    def __call__(self) -> Accumulator[T]:
        return _Buffer(self._func)


class _Buffer(Generic[T]):
    def __init__(self, func: Callable[..., T]) -> None:
        self._func = func
        self._values: list[T] = []

    def push(self, value: T) -> None:
        self._values.append(value)

    @property
    def value(self) -> T:
        if not self._values:
            raise ValueError('Nothing has been pushed')
        return self._func(*self._values)


# --- Aggregation ----------------------------------------------------------------

Contribution = dict[Key, Any]
"""Values at the accumulation keys, per member or combined."""
Table = Mapping[Hashable, Mapping[Key, Any]]
"""Rows of member-key values, by label."""


class Aggregation:
    def __init__(
        self,
        pipeline: sciline.Pipeline,
        *,
        members: Iterable[Key],
        accumulators: Mapping[Key, Callable[[], Accumulator[Any]]],
        outputs: Iterable[Key] = (),
        scheduler: Scheduler | None = None,
    ) -> None:
        """
        Parameters
        ----------
        pipeline:
            The flat pipeline with its parameters set; the aggregation is a snapshot
            of it.
        members:
            The keys supplied per member, the columns of a member table.
        accumulators:
            The accumulation keys, each with a factory for its accumulator.  A key
            that does not depend on the members is not accumulated; finalize computes
            it from the fixed part of the graph.  ``accumulation_keys`` lists the
            keys that are.
        outputs:
            Keys computed by finalize from the accumulation keys.  Omit for an
            aggregation used only for its contributions, such as one of several
            sharing a finalize stage.
        """
        members = tuple(members)
        outputs = tuple(outputs)
        probe = Stage(pipeline, outputs=tuple(accumulators), inputs=members)
        self.accumulation_keys = probe.dynamic_outputs
        self._accumulators = {k: accumulators[k] for k in self.accumulation_keys}
        self.contribute_stage = Stage(
            pipeline,
            outputs=self.accumulation_keys,
            inputs=members,
            scheduler=scheduler,
        )
        self.finalize_stage = (
            Stage(
                pipeline,
                outputs=outputs,
                inputs=self.accumulation_keys,
                scheduler=scheduler,
            )
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

    def accumulators(self) -> dict[Key, Accumulator[Any]]:
        """Fresh accumulators, one per accumulation key, for a caller that pushes."""
        return {k: make() for k, make in self._accumulators.items()}

    def combine(self, contributions: Iterable[Contribution]) -> Contribution:
        """Push each contribution into fresh accumulators and read them."""
        acc = self.accumulators()
        for contribution in contributions:
            for key, a in acc.items():
                a.push(contribution[key])
        return {key: a.value for key, a in acc.items()}

    def finalize(self, contribution: Contribution) -> dict[Key, Any]:
        if self.finalize_stage is None:
            raise ValueError('This aggregation has no outputs')
        return self.finalize_stage(contribution)

    def compute(self, table: Table) -> dict[Key, Any]:
        """Contribute per row, pushing each as it is made; combine; finalize."""
        if not table:
            raise ValueError('The member table is empty')
        warm(*self.stages)
        return self.finalize(
            self.combine(self.contribute(row) for row in table.values())
        )


def compute_members(
    pipeline: sciline.Pipeline, *, members: Iterable[Key], key: Key, table: Table
) -> dict[Hashable, Any]:
    """Per-member value of a key that depends on the member keys."""
    stage = Stage(pipeline, outputs=(key,), inputs=tuple(members))
    return {label: stage(row)[key] for label, row in table.items()}
