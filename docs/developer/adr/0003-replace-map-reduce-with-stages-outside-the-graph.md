# ADR 0003: Replace map/reduce with stages composed outside the graph

- Status: proposed
- Deciders: Simon (proposer); to be discussed with Jan-Lukas, Johannes, Mridul, Neil, Sunyoung
- Date: 2026-09-11

The design document, with the prototype, the survey of current uses, and the rollout plan, is [Map and reduce outside the graph](../architecture-and-design/map-reduce-outside-the-graph.md).

## Context

### What map/reduce does

`Pipeline.map` takes a table of parameter values and copies the part of the graph that depends on them, once per row.
`Pipeline.reduce` adds a node that combines the copies with a function.
The result is still one graph, computed in one call.
The ESS packages use this to process several runs, banks, or files and combine them, usually by grafting the combined value back onto the pipeline:

```python
runs = pd.DataFrame({Filename[SampleRun]: ['run1.nxs', 'run2.nxs']})
workflow[DetectorData] = workflow[DetectorData].map(runs).reduce(func=merge)
workflow.compute(IofQ)
```

Sciline documents this for all users in the parameter-tables guide, so it is not only an ESS feature.

### Why it has to go

- **It blocks the PEP 695 generics.**
  `map` relabels every node that depends on the table at the time it is called.
  With generic providers, most of those nodes do not exist until a target is requested.
  Both prototypes (scipp/sciline#236 and #237, closed unmerged) worked around this, and every breaking change and every open question in them came from map/reduce.
  For example, #237 forbids the `workflow[K] = workflow[K].map(...).reduce(...)` idiom above, and it needs an unreleased cyclebane (scipp/cyclebane#32).
- **It is expensive to keep.**
  This is the second implementation (parameter tables in 23.08, cyclebane in 24.06), and it carries the cyclebane dependency.
  Users cannot write the name of a mapped node, so they need `get_mapped_node_names` and `compute_mapped`, which require pandas.
  `groupby` is announced in the guide but was never implemented, and no ESS package uses `index_names` or `indices`.

### What is needed instead

Several consumers need a related operation that map/reduce cannot express:
run one part of a graph many times with different inputs, combine the results *outside* the graph, then run the rest once.

- `ess.reduce.streaming.StreamProcessor` does this for the chunks of a stream.
  Sciline gives it no way to split a graph, so it prunes branches by setting keys to `None`, grafts subgraphs, and patches provider annotations at run time.
  scipp/sciline#241 asks for proper support.
- The planned reduction service (essapps) needs to cache everything upstream of a cheap parameter, to process members in separate processes and combine them later, and to run two halves of one graph in two processes.

Map/reduce cannot serve these because the loop and the combined value live inside a single `compute` call.
There is no way to push a new member later, to keep per-member results between calls, or to run members in different processes.

If sciline offers this operation, map/reduce can be expressed with it, and the multi-run helpers of esssans, essreflectometry, and bifrost can migrate to it.

## Decision

Remove from sciline: `map`, `reduce`, `index_names`, `indices`, `get_mapped_node_names`, `compute_mapped`, `visualize(compact=)`, and the cyclebane dependency.
Do not add `groupby`.
`constraints=` is removed in the same release, because the PEP 695 generics do not need it.

Add two building blocks that work on an ordinary flat pipeline and add nothing to it.

### `Stage`: a part of a pipeline, from chosen inputs to chosen outputs

```python
stage = Stage(pipeline, inputs=(Filename,), outputs=(Result,))
stage.compute({Filename: 'run1.nxs'})  # -> {Result: ...}
stage.compute({Filename: 'run2.nxs'})  # -> {Result: ...}, calibration not loaded again
```

A stage splits the graph needed for the outputs into two parts:

- The *static* part does not depend on the inputs, for example loading a calibration.
  It is computed once, on first use, and the stage keeps the values that the dynamic part reads (`stage.frontier`).
- The *dynamic* part depends on the inputs.
  It is computed on every call, and nothing of it is kept after the call returns.

An input can be a parameter or an intermediate result; in the second case everything upstream of it is cut off.
A stage is a snapshot: changing the pipeline afterwards does not change the stage.
The snapshot copies the graph but not the parameter values, so a value modified in place, rather than set anew, can change what the stage computes.
`warm(*stages)` computes the static parts of several stages together, so that work they share is done once.

This is the operation that `StreamProcessor` builds by hand and that scipp/sciline#241 asks for.
`Pipeline.provide(key, callable)`, the other request of that issue, is added as well.

### `Aggregation`: combine the results of many members

Terms, using multiple runs as the example:

- A **member** is one run, given as a row of values for the **member keys**, such as `Filename[SampleRun]`.
- An **accumulation key** is a key at which the per-member values are combined, such as `DetectorData`.
- A **contribution** is the dict of values at the accumulation keys for one member.
- An **accumulator** combines contributions: `push(value)` adds one, `value` returns the combination.
  `Buffered(func)` keeps all pushed values and applies an n-ary function, like the `func` of `reduce` today.
  `Reduced(func)` keeps only a running result of a binary function, which saves memory for sums of large arrays.

An aggregation is two stages of one pipeline with accumulators between them:

```text
             contribute                    combine                        finalize
member rows ────────────▶ contributions ─────────────▶ combined value ────────────────▶ outputs
            member keys →                accumulators                 accumulation keys →
            accumulation keys                                         outputs
```

```python
agg = Aggregation(
    pipeline,
    members=(Filename[SampleRun],),
    accumulators={DetectorData: Buffered(merge)},
    outputs=(IofQ,),
)
table = {'r1': {Filename[SampleRun]: 'run1.nxs'}, 'r2': {Filename[SampleRun]: 'run2.nxs'}}
agg.compute(table)

# The same as three steps, which can run at different times or in different processes:
contributions = {label: agg.contribute(row) for label, row in table.items()}
agg.finalize(agg.combine(contributions.values()))
```

A member table is a plain `Mapping[label, Mapping[Key, value]]`; pandas is not needed.
`compute_members` computes one key per member and replaces `compute_mapped`.

### Who owns what

Sciline provides the mechanism: splitting a graph and combining contributions.
An aggregation holds its stages but no contributions and no member table, and it does not react to parameter changes; a changed parameter means a new aggregation.

Everything stateful belongs to the caller: which members exist, which contributions are kept, what a parameter change invalidates, and whether members run in parallel.
In practice:

- Each reduction package returns its own small object in place of the map/reduced pipeline.
  For esssans it holds the pipeline, one aggregation per run type, and the contributions by filename.
  A drop-in replacement for the map/reduced pipeline is a non-goal.
- `StreamProcessor` in ess.reduce becomes a loop over three stages, with its existing accumulators and an object that holds the current context.
  The accumulators fit the `Accumulator` protocol once histogramming moves out of their base class.
- Nested structures, such as banks within runs, are aggregations whose outputs feed other aggregations.
  No graph contains another graph.

### Rollout

`Stage`, `Aggregation`, and the related functions are added in a minor release.
The ESS packages then migrate one at a time while map/reduce still exists.
The removal comes last, in a major release.
Users who depend on map/reduce and do not need the new generics can stay on the last release before the removal.
Keeping the old `Pipeline` in a separate namespace would serve them equally, but nobody intends to maintain it, so pinning is the offer.

## Alternatives considered

- **Keep map/reduce and land the generics on top.**
  Two prototypes tried this and did not resolve the problems described above.
- **Keep map/reduce, deprecated, next to the new building blocks.**
  This keeps cyclebane, still blocks the generics, and leaves two mechanisms for one job.
  The staged rollout gives users the same transition period without keeping both.
- **Let an aggregation turn back into a pipeline** (`as_pipeline()`), so that the `with_*` helpers could keep returning a pipeline.
  Prototyped and dropped.
  It was the only way two aggregations could be combined, so it became the real mechanism instead of a convenience.
  It also fixed the member table at construction and hid a loop inside a provider, with no per-member errors or progress.
- **Let the aggregation hold state** (member tables, kept contributions, rules for what a parameter change invalidates).
  Prototyped and dropped.
  It rebuilt the map/reduced pipeline in a different form, and it was hard to predict what a parameter change would recompute.
- **An n-ary combine function per accumulation key**, as `reduce(func=)` takes today, instead of accumulators.
  This forces holding all contributions in memory before combining.
  `Buffered` keeps the n-ary function for the common case.
- **A generic object that wires stages and accumulators together.**
  The existing uses disagree on when to push and what to reset (stream chunks, context updates, table aggregations), and such an object would be nested workflows under a new name.
  Deferred: if the package objects end up repeating the same code, that code is its starting point.
- **Nested workflows** (a graph inside a node).
  Rejected earlier because parameters had to be passed between levels and the inner graph was hidden.
  Stages are cut from one flat graph, so neither problem arises.

## Consequences

### Positive

- Sciline drops map/reduce and cyclebane, and the PEP 695 generics can land.
- Users outside ESS keep a documented replacement; the parameter-tables guide becomes a guide on stages and aggregations.
- `StreamProcessor` is expected to lose its graph manipulation and keep only its policy.
  The prototype runs this shape in a test; the rewrite of the real class has not been done.
- The same terms (stage, accumulator, accumulation key, contribution, contribute/combine/finalize) apply in sciline, ess.reduce, and essapps.
  They follow Beam, Flink, and Spark; `fold` was avoided because it means reshaping in scipp.
- Every parameter is set on one flat pipeline, and everything held is a plain object that the caller can inspect, clear, or serialize.
- The prototype reproduces the LoKI multi-run reduction: identical results and the same number of provider calls as map/reduce.
  Adding a run after a compute costs only that run's contribution.

### Negative

- Breaking for the `with_*` helpers in esssans, essreflectometry, and bifrost, for `ess.reduce.parameter_mappers` and the widgets built on it, for essreflectometry's `BatchProcessor`, for notebooks, and for the bifrost bank fold in esslivedata.
- The widgets need one interface across the package objects.
  Whether that is a base class or one generic object is decided when the second package migrates.
- Parallelism over members is the caller's job.
  With map/reduce, the dask scheduler ran members in threads for free; on LoKI the map/reduce reference is about 1.7 s faster than the prototype for this reason.
- A map/reduce inside the per-member work of another one (the pixel masks in esssans) is not an aggregation; it becomes a list parameter and a provider.
- A key in `accumulators` that does not depend on the members is computed by finalize, without an error.
  `agg.accumulation_keys` shows which keys are really accumulated, and package tests should check it.
- Stages keep their static values, and package objects keep contributions (binned events for some workflows), so package objects need a way to clear them.
- Visualization of mapped pipelines (`compact=`) goes and needs a replacement that shows stages and accumulators.
  Progress reporting through `Stage` is not implemented yet.
- networkx becomes a direct dependency; today it comes through cyclebane.

Not yet done: the `StreamProcessor` rewrite against its streaming and visualize tests.
Not yet decided: whether the breaking change ships as a major release or as `sciline.v2` (recommendation: major release).
