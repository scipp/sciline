"""Prototype: map/reduce outside the graph.

Two concepts built on a flat sciline pipeline, without cyclebane, map, or reduce:

Stage
    The part of a pipeline from a set of *input* keys to a set of *output* keys.
    Everything the outputs need that does not depend on the inputs is computed once,
    when the stage is built, and cached.  Calling the stage supplies values for the
    inputs and computes only what lies downstream of them.  An input may be a leaf
    (a parameter) or an intermediate node; in both cases its own providers and
    ancestors are cut off.

Fold
    A member table (one row per member, one column per key) and a *cut*: keys at
    which per-member values are combined.  Built from two stages of the same flat
    pipeline: ``contribute`` (member keys -> cut keys) and ``finalize`` (cut keys ->
    outputs).  ``combine`` is a per-key callable over all member values.

Nothing here adds nodes to the author's graph or hides a parameter: both objects are
derived from the flat pipeline at the time they are built, and the pipeline itself
is never modified.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import sciline
from sciline._provider import ArgSpec, Provider
from sciline._utils import key_name
from sciline.handler import HandleAsComputeTimeException
from sciline.scheduler import NaiveScheduler, Scheduler

Key = Hashable
Graph = dict[Key, Provider]


def _task_graph(
    pipeline: sciline.Pipeline, outputs: Sequence[Key], inputs: Sequence[Key]
) -> Graph:
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
        graph = _task_graph(pipeline, self._outputs, self._inputs)
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
        return tuple(o for o in self._outputs if o in self._dynamic_graph)

    def __call__(self, values: Mapping[Key, Any]) -> dict[Key, Any]:
        if set(values) != set(self._inputs):
            raise ValueError(f'Expected values for {self._inputs}, got {tuple(values)}')
        graph: Graph = dict(self._dynamic_graph)
        for k, v in values.items():
            graph[k] = Provider.parameter(v)
        for k, v in self.static.items():
            graph[k] = Provider.parameter(v)
        return _compute(graph, self._outputs, self._scheduler)


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


Combine = Callable[..., Any]


@dataclass(frozen=True)
class Contribution:
    """Per-member (or combined) values at the cut keys."""

    values: dict[Key, Any]


class Fold:
    def __init__(
        self,
        pipeline: sciline.Pipeline,
        *,
        members: Mapping[Key, Sequence[Any]] | Any,  # Any: pandas.DataFrame
        at: Mapping[Key, Combine],
        outputs: Iterable[Key] | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        pipeline = pipeline.copy()
        self._members = _rows(members)
        if not self._members:
            raise ValueError('A fold needs at least one member')
        self._at = dict(at)
        self._outputs = tuple(outputs) if outputs is not None else _sinks(pipeline)
        self._scheduler = scheduler
        member_keys = tuple(next(iter(self._members.values())).keys())
        self._contribute = Stage(
            pipeline, outputs=tuple(self._at), inputs=member_keys, scheduler=scheduler
        )
        # A cut key that does not depend on the members is not folded; finalize
        # computes it itself from the fixed part of the graph.
        self._cut = self._contribute.dynamic_outputs
        self._finalize_stage: Stage | None = None
        self._pipeline = pipeline
        # Nodes that depend on the member keys alone, not on the static frontier,
        # are computed once per member and held at the member frontier: the same
        # propagation StreamProcessor does for its static inputs, per member.
        dynamic = self._contribute._dynamic_graph
        member_only: set[Key] = set()
        changed = True
        while changed:
            changed = False
            for key, provider in dynamic.items():
                if key not in member_only and all(
                    arg in member_keys or arg in member_only
                    for arg in provider.arg_spec.keys()
                ):
                    member_only.add(key)
                    changed = True
        self._member_only_graph = {k: dynamic[k] for k in member_only}
        self._member_frontier = tuple(
            {
                arg
                for k, p in dynamic.items()
                if k not in member_only
                for arg in p.arg_spec.keys()
                if arg in member_only or arg in member_keys
            }
            | {k for k in self._cut if k in member_only}
        )
        self._per_call_graph = {k: p for k, p in dynamic.items() if k not in member_only}
        self._member_values: dict[Hashable, dict[Key, Any]] = {}

    def _member_frontier_values(self, member: Hashable) -> dict[Key, Any]:
        if member not in self._member_values:
            graph: Graph = dict(self._member_only_graph)
            for k, v in self._members[member].items():
                graph[k] = Provider.parameter(v)
            self._member_values[member] = _compute(
                graph, self._member_frontier, self._contribute._scheduler
            )
        return self._member_values[member]

    def _contribution(self, member: Hashable, frontier_values: Mapping[Key, Any]) -> Contribution:
        graph: Graph = dict(self._per_call_graph)
        for k, v in {**frontier_values, **self._member_frontier_values(member)}.items():
            graph[k] = Provider.parameter(v)
        return Contribution(_compute(graph, self._cut, self._contribute._scheduler))

    @property
    def _finalize(self) -> Stage:
        if self._finalize_stage is None:
            self._finalize_stage = Stage(
                self._pipeline,
                outputs=self._outputs,
                inputs=self._cut,
                scheduler=self._scheduler,
            )
        return self._finalize_stage

    @property
    def cut(self) -> tuple[Key, ...]:
        return self._cut

    @property
    def members(self) -> dict[Hashable, dict[Key, Any]]:
        return self._members

    def contribute(self, member: Hashable) -> Contribution:
        return self._contribution(member, self._contribute.static)

    def combine(self, contributions: Sequence[Contribution]) -> Contribution:
        return Contribution(
            {k: self._at[k](*(c.values[k] for c in contributions)) for k in self._cut}
        )

    def finalize(self, contribution: Contribution) -> dict[Key, Any]:
        return self._finalize(contribution.values)

    def compute(self, key: Key | None = None) -> Any:
        combined = self.combine([self.contribute(m) for m in self._members])
        results = self.finalize(combined)
        return results if key is None else results[key]

    def compute_members(self, key: Key) -> dict[Hashable, Any]:
        """Per-member value of a key upstream of or at the cut."""
        stage = Stage(self._pipeline, outputs=(key,), inputs=self._contribute.inputs)
        return {m: stage(row)[key] for m, row in self._members.items()}

    def as_pipeline(self) -> sciline.Pipeline:
        """The flat pipeline with the cut keys provided by this fold.

        The synthesized providers take the contribute stage's static frontier as
        their inputs, so a change to any parameter upstream of the cut reruns the
        fold, and the keys inside the fold (member keys and what depends on them
        only) are no longer part of the graph.
        """
        fold = self
        frontier = fold._contribute.frontier
        member_keys = fold._contribute.inputs

        combined_type = type(
            f'Fold[{", ".join(key_name(k) for k in fold._cut)}]', (_Combined,), {}
        )

        def combined_contribution(*frontier_values: Any) -> _Combined:
            stage_values = dict(zip(frontier, frontier_values, strict=True))
            contributions = [fold._contribution(m, stage_values) for m in fold._members]
            return combined_type(fold.combine(contributions).values)

        pipeline = fold._pipeline.copy()
        pipeline.insert(
            _with_positional_args(combined_contribution, frontier, combined_type)
        )
        for key in fold._cut:
            pipeline.insert(_selector(combined_type, key))
        # Drop what only the members needed: the member keys and their exclusive
        # descendants are no longer reachable from any output.
        pruned = sciline.Pipeline()
        for key in fold._outputs:
            pruned[key] = pipeline[key]
        return pruned


class _Combined(dict):
    """Cut values of a fold, as one graph node."""


def _with_positional_args(func: Callable[..., Any], args: Sequence[Key], ret: Key) -> Callable[..., Any]:
    # sciline reads a provider's signature with getfullargspec, so the synthesized
    # provider needs real named parameters. (v2 would insert a Provider directly.)
    names = [f'a{i}' for i in range(len(args))]
    namespace: dict[str, Any] = {'func': func}
    exec(f"def provider({', '.join(names)}): return func({', '.join(names)})", namespace)
    provider = namespace['provider']
    provider.__annotations__ = dict(zip(names, args, strict=True)) | {'return': ret}
    return provider


def _selector(combined_type: type, key: Key) -> Callable[[_Combined], Any]:
    def select(combined: _Combined) -> Any:
        return combined[key]

    select.__annotations__ = {'combined': combined_type, 'return': key}
    return select


def _sinks(pipeline: sciline.Pipeline) -> tuple[Key, ...]:
    graph = pipeline.underlying_graph
    return tuple(n for n, d in graph.out_degree() if d == 0)


def _rows(members: Any) -> dict[Hashable, dict[Key, Any]]:
    if hasattr(members, 'iterrows'):  # pandas.DataFrame, columns are keys
        return {idx: dict(row.items()) for idx, row in members.iterrows()}
    keys = list(members)
    n = len(members[keys[0]])
    return {i: {k: members[k][i] for k in keys} for i in range(n)}
