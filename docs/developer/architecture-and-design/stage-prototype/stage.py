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
    A driver for the table-fold shape: per group of member keys, a table with one
    row per member and a set of cut keys with an n-ary combine per key.  Members and
    parameters can be changed after construction; contributions are held per member,
    so adding a member costs one contribution.  Built from stages of one flat
    pipeline: per group a member stage (member keys to the member frontier, the
    nodes that read nothing but the members) and a contribute stage (member frontier
    to the cut), and one finalize stage (all cuts to the outputs).  ``contribute``,
    ``combine``, and ``finalize`` are also exposed separately, so that the three can
    run in different processes with the contribution serialized between them.

Nothing here adds nodes to the author's graph or hides a parameter: every object is
derived from the flat pipeline at the time it is built.  ``Stage`` needs the graph
and belongs in sciline; the rest is policy and belongs next to ``StreamProcessor``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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
"""Values at the cut keys of one group, per member or combined."""


@dataclass
class _Stages:
    cut: tuple[Key, ...]
    member: Stage  # member keys -> member frontier; reads nothing but the members
    contribute: Stage  # member frontier -> cut; its static part is the outer frontier


@dataclass
class _Group:
    keys: tuple[Key, ...]
    at: dict[Key, Combine]
    members: dict[Hashable, dict[Key, Any]] = field(default_factory=dict)
    stages: _Stages | None = None
    member_values: dict[Hashable, dict[Key, Any]] = field(default_factory=dict)
    contributions: dict[Hashable, Contribution] = field(default_factory=dict)


class Fold:
    def __init__(
        self,
        pipeline: sciline.Pipeline,
        *,
        over: Mapping[Key | tuple[Key, ...], Mapping[Key, Combine]],
        outputs: Iterable[Key] | None = None,
        keep_members: bool = False,
        scheduler: Scheduler | None = None,
    ) -> None:
        """
        Parameters
        ----------
        pipeline:
            The flat pipeline; a copy is taken, later changes go through ``fold[key]``.
        over:
            Per group of member keys, the cut keys with their n-ary combine function.
            A group is identified by its member keys, which are the columns of its
            member table.  A member key of one group is an ordinary parameter of the
            others.  A cut key that does not depend on the group's member keys is not
            folded; finalize computes it from the fixed part of the graph.
        outputs:
            Keys computed by finalize; the pipeline's sinks if omitted.
        keep_members:
            Hold the member frontier, what the graph computes from a member alone
            (for a run: the loaded data), across parameter changes, so that a change
            upstream of the cut does not reload members.  Costs memory.
        """
        self._pipeline = pipeline.copy()
        self._outputs = tuple(outputs) if outputs is not None else _sinks(pipeline)
        self._keep_members = keep_members
        self._scheduler = scheduler
        self._groups = [
            _Group(keys=k if isinstance(k, tuple) else (k,), at=dict(at))
            for k, at in over.items()
        ]
        member_keys = [k for g in self._groups for k in g.keys]
        if len(set(member_keys)) != len(member_keys):
            raise ValueError('A key may be a member key of one group only')
        self._finalize: Stage | None = None

    # -- structure, derived from the graph ---------------------------------------

    def _stages(self, group: _Group) -> _Stages:
        if group.stages is None:
            probe = Stage(self._pipeline, outputs=tuple(group.at), inputs=group.keys)
            cut = probe.dynamic_outputs
            whole = Stage(self._pipeline, outputs=cut, inputs=group.keys)
            if whole.frontier:
                # Nodes downstream of the outer frontier are computed per contribution;
                # what they read that is not static is the member frontier.
                outer = Stage(self._pipeline, outputs=cut, inputs=whole.frontier)
                dynamic = set(whole.dynamic)
                member_frontier = tuple(k for k in outer.frontier if k in dynamic)
            else:
                member_frontier = cut
            group.stages = _Stages(
                cut=cut,
                member=self._stage(outputs=member_frontier, inputs=group.keys),
                contribute=self._stage(outputs=cut, inputs=member_frontier),
            )
        return group.stages

    def _stage(self, *, outputs: Iterable[Key], inputs: Iterable[Key]) -> Stage:
        return Stage(self._pipeline, outputs=outputs, inputs=inputs, scheduler=self._scheduler)

    @property
    def cut(self) -> tuple[Key, ...]:
        """The cut keys of all groups that depend on their members."""
        return tuple(k for g in self._groups for k in self._stages(g).cut)

    def _finalize_stage(self) -> Stage:
        if self._finalize is None:
            self._finalize = self._stage(outputs=self._outputs, inputs=self.cut)
        return self._finalize

    def _all_stages(self) -> list[Stage]:
        stages = [self._stages(g) for g in self._groups]
        return [*(s.member for s in stages), *(s.contribute for s in stages), self._finalize_stage()]

    def _group_of(self, keys: Iterable[Key]) -> _Group:
        keys = frozenset(keys)
        for group in self._groups:
            if frozenset(group.keys) == keys:
                return group
        raise KeyError(f'No group with member keys {tuple(keys)}')

    def _group_of_cut(self, keys: Iterable[Key]) -> _Group:
        keys = frozenset(keys)
        for group in self._groups:
            if frozenset(self._stages(group).cut) == keys:
                return group
        raise KeyError(f'No group with cut keys {tuple(keys)}')

    # -- parameters and members --------------------------------------------------

    def __setitem__(self, key: Key, value: Any) -> None:
        """Set a parameter; what depends on it is rebuilt, what does not is kept."""
        for group in self._groups:
            if key in group.keys:
                raise ValueError(f'{key} is a member key; use set_members')
        self._pipeline[key] = value
        for group in self._groups:
            if group.stages is None:
                continue
            if key in group.stages.member.keys:
                group.member_values.clear()
            if key in group.stages.contribute.keys:
                group.contributions.clear()
                group.stages = None
        if self._finalize is not None and key in self._finalize.keys:
            self._finalize = None

    def set_members(self, table: Any) -> None:
        """Set the member table of the group whose member keys are the columns.

        A row is a member; the row index is the member label.  Held values of a
        member whose label and row are unchanged are kept, so that adding a member
        costs one contribution.
        """
        rows = _rows(table)
        if not rows:
            raise ValueError('The member table is empty')
        group = self._group_of(next(iter(rows.values())))
        kept = {
            label
            for label, row in rows.items()
            if label in group.members and _same_row(group.members[label], row)
        }
        group.members = rows
        for held in (group.member_values, group.contributions):
            for label in list(held):
                if label not in kept:
                    del held[label]

    @property
    def members(self) -> dict[tuple[Key, ...], dict[Hashable, dict[Key, Any]]]:
        return {g.keys: dict(g.members) for g in self._groups}

    def clear(self) -> None:
        """Drop held member values and contributions."""
        for group in self._groups:
            group.member_values.clear()
            group.contributions.clear()

    # -- the three entry points --------------------------------------------------

    def contribute(self, row: Mapping[Key, Any]) -> Contribution:
        """The contribution of one member, given as a row of its group's table."""
        stages = self._stages(self._group_of(row))
        warm(stages.member, stages.contribute)
        return stages.contribute(stages.member(dict(row)))

    def combine(self, contributions: Sequence[Contribution]) -> Contribution:
        """Combine contributions of one group with the group's per-key functions."""
        group = self._group_of_cut(contributions[0])
        return {k: group.at[k](*(c[k] for c in contributions)) for k in self._stages(group).cut}

    def finalize(self, contributions: Iterable[Contribution]) -> dict[Key, Any]:
        """The outputs, from one combined contribution per group."""
        merged: Contribution = {}
        for contribution in contributions:
            merged |= contribution
        return self._finalize_stage()(merged)

    # -- driver ------------------------------------------------------------------

    def _contribution(self, group: _Group, label: Hashable) -> Contribution:
        stages = self._stages(group)
        values = group.member_values.get(label)
        if values is None:
            values = stages.member(group.members[label])
            if self._keep_members:
                group.member_values[label] = values
        return stages.contribute(values)

    def compute(self, key: Key | None = None) -> Any:
        """Contribute what is not held, combine per group, finalize."""
        empty = [g.keys for g in self._groups if not g.members]
        if empty:
            raise ValueError(f'No members set for {empty}')
        # Static work shared between stages, such as a file every group reads, once.
        warm(*self._all_stages())
        combined = []
        for group in self._groups:
            for label in group.members:
                if label not in group.contributions:
                    group.contributions[label] = self._contribution(group, label)
            combined.append(self.combine(list(group.contributions.values())))
        results = self.finalize(combined)
        return results if key is None else results[key]

    def compute_members(self, key: Key) -> dict[Hashable, Any]:
        """Per-member value of a key that depends on one group's member keys."""
        ancestors = Stage(self._pipeline, outputs=(key,), inputs=()).keys
        groups = [g for g in self._groups if ancestors & set(g.keys)]
        if len(groups) != 1:
            raise ValueError(f'{key} depends on {len(groups)} groups, expected one')
        (group,) = groups
        stage = self._stage(outputs=(key,), inputs=group.keys)
        return {label: stage(row)[key] for label, row in group.members.items()}


def _sinks(pipeline: sciline.Pipeline) -> tuple[Key, ...]:
    graph = pipeline.underlying_graph
    return tuple(n for n, d in graph.out_degree() if d == 0)


def _rows(table: Any) -> dict[Hashable, dict[Key, Any]]:
    if hasattr(table, 'iterrows'):  # pandas.DataFrame, columns are keys
        return {idx: dict(row.items()) for idx, row in table.iterrows()}
    keys = list(table)
    n = len(table[keys[0]])
    return {i: {k: table[k][i] for k in keys} for i in range(n)}


def _same_row(a: Mapping[Key, Any], b: Mapping[Key, Any]) -> bool:
    # Identity, or equality for plain hashable values such as filenames; arrays
    # compare elementwise and are never considered equal here.
    return all(
        x is y or (isinstance(x, Hashable) and type(x) is type(y) and x == y)
        for x, y in ((a[k], b[k]) for k in a)
    )
