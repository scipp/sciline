# ADR 0003: Replace map/reduce with stages composed outside the graph

- Status: proposed
- Deciders: Simon (proposer); to be discussed with Jan-Lukas, Johannes, Mridul, Neil, Sunyoung
- Date: 2026-09-11

Design document with prototype and validation: [Map and reduce outside the graph](../architecture-and-design/map-reduce-outside-the-graph.md).

## Context

`Pipeline.map` and `Pipeline.reduce` put a loop over members and a combine step inside the task graph.
This has been implemented twice, as parameter tables (23.08) and on cyclebane (24.06).
In the PEP 695 generics work (scipp/sciline#236, #237) every breaking change and every remaining open question came from map/reduce: forward chaining and its seeding, the `reduce(key=)` guard, the node-name uniqueness break that forbids the documented `pipeline[C] = pipeline[C].map(...).reduce(...)` idiom, and a dependency on an unreleased cyclebane.
Users cannot write a mapped key; they need `get_mapped_node_names` and `compute_mapped`, which depend on pandas and reach into private state.
`groupby` was never exposed, and `index_names`/`indices` are unused downstream.

At the same time, three consumers need an operation map/reduce cannot express: run one part of a graph repeatedly with some values supplied per call, combine the results outside, run the rest once.

- `ess.reduce.streaming.StreamProcessor` (1081 lines) builds this partition by hand, pruning branches by assigning `None`, grafting subgraphs, and patching provider annotations at runtime; scipp/sciline#241 records the requirements: no graph rebuild per value, explicit lifetime of supplied values, culling of a supplied key's ancestors.
- The essapps architecture needs the same partition in three places: the warm workflow (cache the nodes upstream of the cheap parameters), the declared additive combine (contribute, combine, finalize), and split workflows across processes.
- Every ESS reduction package folds runs at one or more keys and grafts the result back onto the pipeline (`with_sample_runs` and siblings), which `ess.reduce.parameter_mappers` and the GUI depend on.

## Decision

Remove `map`, `reduce`, `groupby`, `index_names`, `indices`, `get_mapped_node_names`, `compute_mapped`, `constraints=`, and the cyclebane dependency from sciline, and make the graph partition the primitive:

1. **`Stage`**, in sciline.
   The part of a pipeline from named input keys to named output keys.
   The static part, everything the outputs need that does not depend on the inputs, is computed once and held at its frontier; a call supplies exactly the inputs and computes only the dynamic part.
   An input may be a parameter or an intermediate node; an output that is an input is passed through.
   A stage is a snapshot of the pipeline.
   `warm(*stages)` computes the static parts of several stages of one pipeline in one run, so shared static work is done once.
   `Stage` needs the concrete graph and `Provider` and is the only piece that belongs in sciline.

2. **Connectors** between stages, in ess.reduce: objects with `push`, `value`, `clear`.
   The type of the connector says why the graph is cut there: a reducer (the existing accumulators, or an n-ary combine) where members are combined; a `Forwarder` where a context is held between changes; a dict by member label where contributions are kept by whoever loops over members.
   Every held value is an object the caller can inspect, clear, serialize, or place on a process boundary.

3. **`Fold`**, in ess.reduce: two stages of one flat pipeline, contribute (member keys to cut keys) and finalize (cut keys to outputs, optional), with an n-ary combine per cut key between them.
   It holds nothing.
   Parameters are set on the pipeline before it is built; a changed parameter means a new fold.
   `contribute`, `combine`, `finalize` are the entry points essapps calls in separate processes; `compute(table)` is the loop for a caller that holds nothing.
   A cut key that does not depend on the members is not folded.

4. **Drivers and package objects** own topology, state, and policy.
   `StreamProcessor` becomes three stages, a forwarder, and accumulators.
   Each reduction package returns its own object instead of a map/reduced pipeline: esssans's holds the pipeline, one fold per run type, the finalize stage over both cuts, and the contributions by filename, and decides with one test (`key in fold.contribute_stage.keys`) what a parameter change keeps.
   Hierarchy (banks over runs, angle groups inside a run) is composition of these objects; nothing nests inside a graph.

A drop-in replacement for the map/reduced pipeline is a non-goal.

## Alternatives considered

- **Keep map/reduce and fix it.**
  Two implementations and the PEP 695 branch show the cost; the open questions do not close.
- **A bridge back into the pipeline** (`as_pipeline()`, synthesizing providers for the cut keys) so that `with_*` helpers keep returning a pipeline.
  Prototyped and dropped: it was the only way two folds composed, so it was the mechanism rather than a bridge; it fixed the member table at construction; and it hid a loop in a provider with no per-member errors, progress, or visualization.
- **A `Stage` with tiers of inputs and a hold policy per tier**, deriving all boundaries itself.
  Same information as connectors, held inside one class and putting policy into sciline.
- **A `Fold` that sets parameters and holds state** (member tables with groups, contributions kept by a row-equality heuristic, a held per-member frontier, invalidation rules in `__setitem__`).
  Prototyped and stripped: it merged `Pipeline`'s job with orchestration, reproduced the map/reduced pipeline, and was hard to reason about; on LoKI the held frontier bought nothing because the wavelength conversion reads a parameter.
- **A generic network object** holding stages and connectors and routing pushes.
  The real shapes disagree on the push policy (eager chunks, context updates that keep accumulators, lazy table folds, `clear` semantics), and it is the nested-workflow shape one level up.
  Deferred: if the package objects turn out to repeat the same thirty lines, that object is their generalisation.

## Consequences

### Positive

- Sciline loses map/reduce, cyclebane, and the guards; the PEP 695 generics land without their open questions.
  16 map/reduce tests go, 4 are trimmed, the rest port.
- `StreamProcessor` shrinks to its policy, meeting scipp/sciline#241 by construction.
- One vocabulary across sciline, ess.reduce, and essapps: stage, cut, contribution, contribute/combine/finalize.
- Validated on the LoKI multi-run reduction: results identical to the map/reduce reference, equal provider call counts, 6.9 s against 7.2 s under the same scheduler; adding a run costs one contribution.
- Every parameter is set on the flat pipeline and reaches every stage; what is held is visible as objects.

### Negative

- Breaking for every `with_*` caller, `parameter_mappers`, `BatchProcessor`, notebooks, and esslivedata's bifrost bank fold; the ESS monorepo migrates in one PR, esslivedata pins until it follows.
- The GUI needs one protocol across package objects; whether that is a base class or one generic object from a registry of member key to cut is decided when the second package is migrated.
- Parallelism over members, which one graph under the dask scheduler gave for free, becomes the driver's responsibility.
- A fold whose cut sits inside another fold's per-member work (esssans pixel masks) is no longer a fold; it becomes a list parameter and a provider.
- Not yet done: the `StreamProcessor` rewrite against its 35 tests; the choice between `sciline.v2` and a major release (recommendation: major release).
