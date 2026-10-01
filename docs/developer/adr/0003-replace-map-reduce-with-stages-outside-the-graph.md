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
- The planned reduction service (essapps) needs to cache everything upstream of the parameters a user tunes, to add a run to a held sum without reducing the others again, and to reduce the runs of a sum in separate processes and combine them later.

Map/reduce cannot serve these because the loop and the combined value live inside a single `compute` call.
There is no way to push a new member later, to keep per-member results between calls, or to run members in different processes.

If sciline offers this operation, map/reduce can be expressed with it, and the multi-run helpers of esssans, essreflectometry, and bifrost can migrate to it.

## Decision

Remove from sciline: `map`, `reduce`, `index_names`, `indices`, `get_mapped_node_names`, `compute_mapped`, `visualize(compact=)`, and the cyclebane dependency.
Do not add `groupby`.
`constraints=` is removed in the same release, because the PEP 695 generics do not need it.

Add building blocks that work on an ordinary flat pipeline and add nothing to it: `Stage` with `warm`, accumulators, and `enclose` for nested loops.

### `Stage`: a part of a pipeline, from chosen inputs to chosen outputs

```python
stage = Stage(pipeline, inputs=(Filename,), outputs=(Result,))
stage.compute({Filename: 'run1.nxs'})  # -> {Result: ...}
stage.compute({Filename: 'run2.nxs'})  # -> {Result: ...}, calibration not loaded again
```

A stage splits the graph needed for the outputs into two parts:

- The *static* part does not depend on the inputs, for example loading a calibration.
  It is computed once, on first use, and the stage keeps the values that the dynamic part reads, plus outputs that do not depend on the inputs (`stage.frontier`).
- The *dynamic* part depends on the inputs.
  It is computed on every call, and nothing of it is kept after the call returns.

An input can be a parameter or an intermediate result; in the second case everything upstream of it is cut off.
A stage is a snapshot: changing the pipeline afterwards does not change the stage.
The snapshot copies the graph but not the parameter values, so a value modified in place, rather than set anew, can change what the stage computes.
`warm(*stages)` computes the static parts of several stages together, so that work they share is done once.

This is the operation that `StreamProcessor` builds by hand and that scipp/sciline#241 asks for.
`Pipeline.provide(key, callable)`, the other request of that issue, is added as well.

### Accumulators: combine the results of many members

Terms, using multiple runs as the example:

- A **member** is one run, given by the values of the inputs of a stage, such as `Filename[SampleRun]`.
- An **accumulation key** is a key at which the per-member values are combined, such as `DetectorData`.
- A **contribution** is the dict of values at the accumulation keys for one member.
- An **accumulator** combines the values of all members at one accumulation key: `push(value)` adds one member's value, `value` returns the combination (`Accumulator` protocol).
  `Buffered(func)` keeps all pushed values and applies an n-ary function, like the `func` of `reduce` today.
  `Reduced(func)` keeps only a running result of a binary function, which saves memory for sums of large arrays.
- The **driver** is the loop that calls the stages and pushes into the accumulators.

The replacement for `map(...).reduce(...)` is a loop over a stage with accumulators, and a stage for the rest:

```python
contribute = Stage(pipeline, inputs=(Filename[SampleRun],), outputs=(DetectorData,))
finalize = Stage(pipeline, inputs=(DetectorData,), outputs=(IofQ,))
acc = Buffered(merge)()
for run in ['run1.nxs', 'run2.nxs']:
    acc.push(contribute.compute({Filename[SampleRun]: run})[DetectorData])
finalize.compute({DetectorData: acc.value})
```

Contributing, combining, and finalizing can run at different times or in different processes, since a contribution is a plain dict; the same loop without the accumulator replaces `compute_mapped`.

### `enclose`: the stages of nested loops

Banks within runs, or chunks of a stream within a context, are loops in loops.
The inner level reads values that depend on the outer level only, such as a run's monitor, and those should be computed once per outer iteration.
A stage built for the inner loop holds these values at its frontier.
`enclose` puts the stage inside a loop over the outer inputs, and derives the boundary from the graph:

```python
bank_stage = Stage(pipeline, inputs=(Bank,), outputs=(Numerator, Denominator))
run_stage, bank_stage = enclose(pipeline, [bank_stage], inputs=(Filename,))
final_stage = Stage(pipeline, inputs=(Numerator, Denominator), outputs=(IofQ,))
```

The run stage computes, once per run, the values that the bank stage held and that depend on `Filename`.
The rebuilt bank stage takes these *forwarded values* as inputs.
Nested loops are built from the inside out: enclosing the result again adds a further outer loop, so each value is computed by the deepest loop whose inputs it depends on.
`enclose` returns plain stages.
The driver writes the loops and holds the values between them:

```python
for filename in filenames:
    held = run_stage.compute({Filename: filename})
    for bank in banks:
        out = bank_stage.compute({**held, Bank: bank})
        ...  # push out[Numerator] and out[Denominator] into accumulators
```

`enclose` raises an error where a combined value would otherwise be silently wrong: an output of a stage that does not vary in that stage (a run-level key pushed once per bank).

### Who owns what

Sciline provides the mechanism: cutting a graph into stages and combining the values of members.
Stages hold the values at their frontier but no contributions and no member list, and they do not react to parameter changes; a changed parameter means new stages.

Everything stateful belongs to the driver: which members exist, which contributions are kept, what a parameter change invalidates, and whether members run in parallel:

- Each reduction package returns its own small object in place of the map/reduced pipeline.
  For esssans it holds a contribute stage per run type, a finalize stage, and the contributions by filename.
  A drop-in replacement for the map/reduced pipeline is a non-goal.
- `StreamProcessor` in ess.reduce becomes a driver over stages built with `enclose` (a context stage, with chunk stages and a finalize stage inside its loop), with its existing accumulators and an object that holds the current context.
  The accumulators fit the `Accumulator` protocol once histogramming moves out of their base class.
- Nested structures, such as banks within runs, are loops over stages built with `enclose`; no graph contains another graph.

### Rollout

`Stage`, `warm`, `enclose`, and the accumulators are added in a minor release.
The ESS packages then migrate one at a time while map/reduce still exists.
The removal comes last, in a major release (recommended; a `sciline.v2` namespace is the open alternative, see below).
Users who depend on map/reduce and do not need the new generics can stay on the last release before the removal.
A `sciline.v2` namespace that keeps the old `Pipeline` would serve them equally, but nobody intends to maintain it.

## Alternatives considered

- **Keep map/reduce and land the generics on top.**
  Two prototypes tried this and did not resolve the problems described above.
- **Keep map/reduce, deprecated, next to the new building blocks.**
  This keeps cyclebane, still blocks the generics, and leaves two mechanisms for one job.
  The staged rollout gives users the same transition period without keeping both.
- **An `Aggregation` object** with a contribute stage, accumulators, a finalize stage, and `compute(table)`.
  Prototyped and dropped: no real driver used its finalize stage or `compute`, since the SANS finalize reads two run types and the Bifrost per-run finalize also reads run-level values.
  `compute(table)` invited a flat runs-times-banks table, which counts run-level keys once per bank without an error, and its one guarantee is `Stage.dynamic_outputs`.
- **Choosing the per-run keys of nested loops by hand.**
  Prototyped for LoKI and Bifrost: a key left out is recomputed per bank with a correct result, so the mistake goes unnoticed.
- **`split(pipeline, *parts)`, with one `Part` per loop** that names the part of its enclosing loop as `parent`.
  Prototyped and dropped: `parent` declared the forwarded values only indirectly, which made the API hard to understand.
  It built the same stages as `enclose`.
  Since it saw all loops at once, it rejected some mistakes at construction, such as a read from a loop that does not enclose the reader.
  `enclose` builds on the frontier, which users of `Stage` already know.
- **Let an aggregation turn back into a pipeline** (`as_pipeline()`), so that the `with_*` helpers could keep returning a pipeline.
  Prototyped and dropped: it became the only way to combine two aggregations, fixed the members at construction, and hid a loop inside a provider.
- **Let the sciline objects hold state** (member tables, kept contributions, rules for what a parameter change invalidates).
  Prototyped and dropped: it rebuilt the map/reduced pipeline in a different form, and it was hard to predict what a parameter change would recompute.
- **An n-ary combine function per accumulation key**, as `reduce(func=)` takes today, instead of accumulators.
  This forces holding all contributions in memory before combining.
  `Buffered` keeps the n-ary function for the common case.
- **A generic object that wires stages and accumulators together.**
  The existing uses disagree on when to push and what to reset (stream chunks, context updates, loops over runs), and such an object would be nested workflows under a new name.
  Deferred: if the package objects end up repeating the same code, that code is its starting point.
- **Nested workflows** (a graph inside a node).
  Rejected earlier because parameters had to be passed between levels and the inner graph was hidden.
  Stages are cut from one flat graph, so neither problem arises.

## Consequences

### Positive

- Sciline drops map/reduce and cyclebane, and the PEP 695 generics can land.
- Users outside ESS keep a documented replacement; the parameter-tables guide becomes a guide on stages.
- `StreamProcessor` is expected to lose its graph manipulation and keep only its policy.
- The same terms (stage, accumulator, accumulation key, contribution, contribute/combine) apply in sciline, ess.reduce, and essapps.
  Accumulator, contribution, and contribute/combine follow Beam, Flink, and Spark.
- Every parameter is set on one flat pipeline, and everything held is a plain object that the driver can inspect, clear, or serialize.
- The prototype reproduces the LoKI multi-run reduction with identical results and provider calls, and adding a run costs only that run's contribution.
  Prototype drivers, which are not in this repository, gave the results of plain loops for banks or triplets times runs, and a prototype `StreamProcessor` gave those of the real class.
  They were built with `split` (see alternatives), which builds the same stages as `enclose` for these shapes.

### Negative

- Breaking for the `with_*` helpers in esssans, essreflectometry, and bifrost, for `ess.reduce.parameter_mappers` and the widgets built on it, for essreflectometry's `BatchProcessor`, for notebooks, and for the bifrost bank fold in esslivedata.
- The widgets need one interface across the package objects, a base class or one generic object, decided when the second package migrates.
- Parallelism over members is the driver's job; with map/reduce, dask ran members in threads for free (about 1.7 s of 7.5 s in the LoKI validation script).
- A map/reduce inside the per-member work of another one (the pixel masks in esssans) becomes a list parameter and a provider.
- A one-level driver that builds stages by hand must push only `stage.dynamic_outputs`; other keys are counted once per member. For nested loops, `enclose` is needed: a run-level key in a stage with inputs `(Filename, Bank)` is dynamic and still counted once per bank.
- Each loop is inside at most one other: bank-only work is computed once per run and bank, and a `StreamProcessor` context update recomputes all context-derived values (in esslivedata compute only; no accumulators reset).
- `enclose` sees one loop at a time, so some mistakes surface in the driver or in `warm`, not when the stages are built: a stage left out of an `enclose` call, or enclosed twice over the same inputs.
- Nothing detects the esssans background stage reading the masks of the one sample run set on the pipeline; esssans has to decide which run's detector IDs the background masks use.
- esslivedata selects its scheduler by replacing `sciline.task_graph.DaskScheduler`; sciline keeps that working for stages, until it offers a public way.
- Stages keep their held values, and package objects keep contributions (binned events for some workflows), so package objects need a way to clear them.
- Visualization of mapped pipelines (`compact=`) goes; `Stage.visualize` and `visualize_stages` replace it in part.
  Progress reporting through `Stage` is not implemented yet.
- networkx becomes a direct dependency; today it comes through cyclebane.

Not yet done: the `StreamProcessor` rewrite against its streaming and visualize tests.
Not yet decided: whether the breaking change ships as a major release or as `sciline.v2` (recommendation: major release).
