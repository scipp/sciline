# Map and reduce outside the graph

This document belongs to [ADR 0003](../adr/0003-replace-map-reduce-with-stages-outside-the-graph.md).
The ADR explains why `Pipeline.map` and `Pipeline.reduce` should be removed and what replaces them.
This document gives the details: how the replacement behaves, how it covers each current use of map/reduce, which design options were rejected and why, how the prototype was validated, and how the ESS packages migrate.

The reader is assumed to know `sciline.Pipeline` and to have read the ADR.

## 1. How map/reduce is used today

The design has to cover every current use.
A survey of the ESS packages (the scipp/ess monorepo) and esslivedata on 2026-09-11 found these shapes:

| Shape | Where | How values are combined |
|---|---|---|
| Map over a table, reduce at one or more keys, then continue with further providers | esssans runs (four map/reduce pairs per run type), essreflectometry runs (up to seven reduced keys), isissans zoom monitors, bifrost detector triplets (three folds in two workflows), NMX panels and MTZ files, DREAM detectors with a two-column table | events by concatenation, dense data by sum, metadata by picking one value, `DataGroup` by key |
| A reduce that sits inside the per-member work of another map | esssans pixel masks: `DetectorMasks` reads the detector IDs of each sample run. (DREAM has the same fold, but there it does not depend on the run.) | dicts by union |
| Per-member results without a reduce | esssans `with_banks`, LoKI notebook, essreflectometry `BatchProcessor`, a bifrost test | not combined; read with `compute_mapped` |
| Several map/reduces on one pipeline, then one final computation | sample runs and background runs | as above |
| Map/reduce in the part of a `StreamProcessor` that is computed once | esslivedata bifrost: `EmptyDetector` over banks | `_combine_banks` |

Two further observations matter for the design:

- The packages graft the reduced node back onto the pipeline (`workflow[K] = workflow[K].map(df).reduce(func=f)`), so users see an ordinary pipeline afterwards.
  `ess.reduce.parameter_mappers` and the parameter widgets rely on this.
  The design gives this up (see section 10).
- essreflectometry wraps each `reduce` in `try/except`, because a key it reduces may not depend on the mapped key at all.

## 2. Overview

The author of a workflow writes one flat pipeline, as today.
A caller that needs repetition cuts the pipeline into *stages*, calls them from an ordinary Python loop, and places objects between the stages that hold state:

```text
            ┌──────────────┐        ┌─────────────┐        ┌──────────────┐
 inputs ───▶│   stage A    │──push─▶│  connector  │─value─▶│   stage B    │───▶ outputs
 per call   └──────────────┘        └─────────────┘        └──────────────┘
                                 (accumulator or forwarder)
```

Sciline provides the stages and the `Accumulator` protocol with two ready-made accumulators.
It also provides `Aggregation`, which packages the most common arrangement: a stage per member, accumulators, and a final stage.
Everything else, in particular what is kept between calls and when it is discarded, is decided by the code that runs the loop.

Terms used in this document:

| Term | Meaning |
|---|---|
| stage | The part of a pipeline from a set of input keys to a set of output keys (`sciline.Stage`). |
| static part, dynamic part | The part of a stage's graph that does not depend on the inputs, and the part that does. |
| frontier | The keys of the static part that the dynamic part reads, plus outputs that do not depend on the inputs. A stage holds the values at these keys. |
| connector | An object between stages with `push(value)` and `value`. |
| accumulator | A connector that combines all pushed values (`sciline.Accumulator`). |
| forwarder | A connector that holds the last pushed value until the next push. Lives in ess.reduce. |
| member | One repetition, for example one run. Given as a row of values for the *member keys*. |
| accumulation key | A key at which the values of the members are combined. |
| contribution | The values at the accumulation keys for one member, as a dict. |
| driver | The loop that calls stages and pushes into connectors. |
| package object | What a reduction package returns to users in place of today's map/reduced pipeline. It is a driver. |

## 3. `Stage`

```python
stage = Stage(pipeline, inputs=(Filename,), outputs=(Numerator, Denominator))
stage.frontier                        # keys whose values the stage holds
stage.compute({Filename: 'run1.nxs'}) # -> {Numerator: ..., Denominator: ...}
warm(stage_a, stage_b)                # compute the static parts of both in one run
```

### Behaviour

- **Partition.**
  The stage takes the graph of all ancestors of the outputs.
  The inputs and everything that depends on them form the dynamic part; the rest is the static part.
  An input can be a parameter or an intermediate result.
  In both cases its provider and everything upstream of it are removed from the stage.
- **Held values.**
  On first use, the stage computes the values at its frontier and keeps them.
  Nothing else of the static part is kept.
- **Calls.**
  A call must supply a value for each input and nothing else.
  It computes only the dynamic part, reading the held frontier values, and returns the outputs.
  The supplied and intermediate values are released when the call returns.
- **Pass-through.**
  An output that is also an input is returned as supplied.
  This makes a stage from a key to itself the identity, which an aggregation needs when its members are already values at its accumulation key (section 6.3).
- **Snapshot.**
  The stage is built from the task graph of the pipeline at construction time.
  Later changes to the pipeline do not affect it; to change a parameter, build a new stage.
  Parameter values are held by reference, not copied, so a value modified in place, rather than set anew, can change what the stage computes.
- **Introspection.**
  `dynamic` lists the keys that depend on the inputs, and `dynamic_outputs` the outputs among them.
  `keys` lists every key the stage uses.
  A parameter is in `keys` exactly when changing it would change the stage's results; callers use this to decide what to rebuild (section 6.1).
- **`warm(*stages)`.**
  Stages built from one pipeline often share static work, for example a file that each of them reads.
  `warm` computes the static parts of several stages in one scheduler run, so shared intermediate results are computed once and then released.
  Each stage keeps only its own frontier values, as if it had been warmed alone.
- **Threads.**
  A stage can be called from several threads at once; the static part is still computed only once.

### Implementation notes

The stage builds its task graph with `to_task_graph` and `HandleAsComputeTimeException`.
Keys without a value, such as the inputs, thus become nodes that fail only if they are computed, instead of failing when the stage is built.
The scheduler default is the same as for `Pipeline`.

### Why it is in sciline

A stage needs the concrete task graph and the `Provider` objects.
Outside sciline this means private access, which is how `StreamProcessor` works today, and scipp/sciline#241 asks for a public way.

In `StreamProcessor`, a stage replaces `_build_streaming_workflow`, `_FedWorkflow`, `_find_descendants`, `_find_parents`, and the pruning of branches by setting keys to `None`.
It also meets the three requirements of scipp/sciline#241: no graph rebuild per value, a well-defined lifetime for supplied values, and no computation of the ancestors of a supplied key.

## 4. Accumulators

### The protocol

`Accumulator` is a structural protocol: any object with `push(value)` and a `value` property satisfies it.
Sciline provides two factories:

- `Buffered(func)` makes accumulators that keep every pushed value and apply the n-ary function `func` to all of them when `value` is read.
  This is the signature of `reduce(func=)` today, so every existing combine function can be used unchanged.
- `Reduced(func)` makes accumulators that keep only a running result.
  The first pushed value becomes the result, and each further push replaces it by `func(result, value)`.
  `func` must be associative and must not modify its arguments, because the pushed values belong to the caller.
  Since the update is not in place, a sum briefly holds the old result, the new result, and the pushed value.
  An accumulator that updates in place has to be written by the workflow author, who then owns the copy of the first value.
  The accumulators in ess.reduce are such objects.

### Combining combined values

`Aggregation.combine` accepts combined values as well as individual contributions.
This allows combining in groups or as a chain, for example one new member at a time onto a previous result.
It requires two things of an accumulator:

1. What `value` returns can be pushed again.
2. The result does not depend on how the pushes were grouped.

`Buffered` satisfies both if its function is associative, that is `func(func(a, b), c) == func(a, b, c)`.
`Reduced` requires associativity anyway.
An ess.reduce accumulator satisfies both if its `push` accepts what its `value` returns.
The histogramming accumulators must keep this property once `maybe_hist` moves out of their base class (see below).

### Which one to use

For concatenation, buffering and a running result cost about the same memory, roughly twice the total, and `Buffered` is the natural choice.
For a sum of large dense arrays, a running result holds one array instead of one per member, and `Reduced` or a custom in-place accumulator is better.

### Relation to the ess.reduce accumulators

An accumulator in an aggregation over a table and an accumulator in a stream are the same kind of object.
Whether the values come from many members or from one input over time only matters to the driver.
The driver also owns everything else about the values: their lifetime (`clear`), their identity (contributions by label), and whether the result may depend on the order of pushes.

The accumulators of `ess.reduce.streaming` satisfy the protocol once `maybe_hist`, which histograms in the base class `push`, moves to the subclasses that need it.
The base class then adds nothing to the protocol, and a forwarder built on it does not histogram.
Their `clear` method stays an ess.reduce convention for accumulators that are reused between finalizations.
No base class beyond the protocol is shared between sciline and ess.reduce.
`Accumulator` is in sciline because `Aggregation` consumes it; `Forwarder` stays in ess.reduce because nothing in sciline consumes one.

## 5. `Aggregation`

```python
agg = Aggregation(
    pipeline,
    members=(Filename[SampleRun],),
    accumulators={CleanSummedQ[SampleRun, Numerator]: Buffered(merge_contributions), ...},
    outputs=(IofQ,),
)
agg.accumulation_keys           # the keys in `accumulators` that depend on the members
agg.contribute(row)             # one member -> contribution
agg.accumulators()              # new accumulators, for a caller that pushes itself
agg.combine(contributions)      # push into new accumulators, return the combined contribution
agg.finalize(combined)          # accumulation keys -> outputs
agg.compute(table)              # all of the above for a table
compute_members(pipeline, members=(Filename[SampleRun],), key=NormalizedQ[SampleRun, Numerator], table=table)
```

### Behaviour

- **Structure.**
  An aggregation consists of two stages of one pipeline: `contribute_stage` from the member keys to the accumulation keys, and `finalize_stage` from the accumulation keys to the outputs.
  Between them sits one accumulator per accumulation key.
- **Parameters.**
  All parameters are set on the pipeline before the aggregation is built.
  Like a stage, the aggregation is a snapshot.
  A changed parameter means a new aggregation, which costs a walk of the graph but no computation until it is used.
- **Factories, not instances.**
  `accumulators` maps each accumulation key to a factory.
  `combine` and `compute` make new accumulators on every call, so no state survives between calls.
  With instances and a `clear` method, the aggregation would have to either forbid combining in batches over several calls or silently add to old state, and `combine` could not safely run concurrently.
  With factories, whoever calls `accumulators()` owns the instances and decides their lifetime.
- **Keys that do not depend on the members.**
  A key in `accumulators` that does not depend on the member keys is not accumulated.
  The finalize stage computes it from the static part instead, and `accumulation_keys` leaves it out.
  This removes the need for essreflectometry's `try/except`.
  It also means that a key placed wrongly is silently treated as static.
  A consumer that knows which keys must be accumulated, such as a package test or the essapps binding, should compare its list with `accumulation_keys`.
- **No outputs.**
  `outputs` is optional.
  Without outputs there is no finalize stage, which is useful when several aggregations share one finalize stage (section 6.2).
- **Separate steps.**
  `contribute`, `combine`, and `finalize` can be called at different times and in different processes.
  A contribution is a plain dict and can be serialized between them.
- **`compute(table)`.**
  This is the loop for a caller that keeps nothing.
  It warms both stages, contributes row by row, and pushes each contribution before the next one is computed, so the peak memory depends on the accumulators, not on the aggregation.
  It then finalizes.
- **Member table.**
  A table is a `Mapping[label, Mapping[Key, value]]`, which `df.to_dict('index')` produces from a pandas DataFrame.
  Sciline does not depend on pandas, and the labels are chosen by the caller.
- **`compute_members`** computes one key for each row of a table, without combining.
  It replaces `compute_mapped`.

### What is deliberately not in `Aggregation`

- **Grouping.**
  `groupby` is a grouping of the table followed by one aggregation per group.
  The caller can do this in a few lines.
- **Parallelism over members.**
  `contribute` is a plain function call, so the caller can map it over rows with threads, processes, or dask, and push results as they arrive.
  The user guide shows an example.
  This is the one thing that computing everything in one graph provided for free.
- **State.**
  The aggregation holds its stages and nothing else: no contributions, no member table, no rules for parameter changes.
  Section 8.4 explains why.

## 6. Composition

Every use from section 1 is a combination of stages, connectors, and a driver.
This section shows each.

### 6.1 A package object (esssans)

esssans users today call `with_sample_runs` and `with_background_runs`, set parameters, and compute.
To keep this experience, esssans returns its own object instead of a pipeline.
The validation script contains such an object, `SansReduction`, in about 60 lines.
It holds:

- the flat pipeline,
- one aggregation per run type (sample and background), each without outputs,
- one finalize stage from the accumulation keys of both aggregations to the outputs,
- the contributions of each run type, by filename.

Its methods:

- `set_runs(run_type, runs)` records the runs and drops the contributions of runs no longer listed.
- `obj[key] = value` sets the parameter on the pipeline.
  For each aggregation with `key in agg.contribute_stage.keys`, it builds a new aggregation and drops that run type's contributions.
  If `key in finalize.keys`, it builds a new finalize stage.
- `compute()` warms all stages together, contributes the runs that have no contribution yet, combines, and finalizes.

This is the only place where the rules for "set runs, set parameters, compute" live, and they fit on one screen.

### 6.2 Several aggregations, one final stage

Sample runs and background runs are two aggregations on the same pipeline.
One `Stage` takes the accumulation keys of both as inputs and computes the final result.
All three stages are warmed together, so work they share, such as reading a mask file, is done once.

### 6.3 Hierarchy

- **Banks over runs.**
  One aggregation per bank, each on the pipeline with that bank selected, produces a result per bank.
  A second aggregation takes these results as members.
  Its member key is also its accumulation key, so its contribute stage is the pass-through identity from section 3.
- **Groups within a run** (for example angle groups in Bifrost).
  An inner aggregation's output is a member of the outer aggregation.

In both cases, nothing is nested inside a graph.

### 6.4 A reduce inside per-member work (esssans pixel masks)

In esssans, `DetectorMasks` reads the detector IDs of the sample run.
With map/reduce, the graph therefore computes the mask fold once per sample run.
This is not an aggregation, because the combined value is needed inside the work for each member.

Outside the graph it becomes a list parameter, `PixelMaskFilenames`, and two providers: one reads all mask files (static and shared by all runs), and one builds the masks for a given run.
Aggregations are meant for members that are expensive to compute or whose individual results users want to see.
A handful of small files combined by union is neither.

### 6.5 Passing a result from one driver to another

A result enters another driver as an ordinary parameter of the flat pipeline.
For example, the bifrost bank fold in esslivedata becomes:

```python
pipeline[EmptyDetector] = banks.compute(table)[EmptyDetector]
processor = StreamProcessor(pipeline, ...)
```

### 6.6 `StreamProcessor`

`StreamProcessor` is not an aggregation.
Its members (chunks of a stream) cannot be recomputed, its accumulators are reused between finalizations and need not be associative (rolling windows), and a change of context, such as a new detector position, must not discard what was accumulated.

It is three stages and two kinds of connector.
This is also its structure today, but the boundary of the context part is now derived from the graph instead of found by hand:

```python
per_chunk = Stage(pipeline, outputs=accumulator_keys, inputs=dynamic_keys)
context_frontier = Stage(pipeline, outputs=per_chunk.frontier, inputs=context_keys).dynamic_outputs
context_stage = Stage(pipeline, outputs=context_frontier, inputs=context_keys)
chunk_stage = Stage(pipeline, outputs=accumulator_keys, inputs=dynamic_keys + context_frontier)
finalize_stage = Stage(pipeline, outputs=target_keys, inputs=accumulator_keys + context_targets)
context = Forwarder()
accumulators = {key: EternalAccumulator() for key in accumulator_keys}
```

- `set_context` calls the context stage and pushes the result into the forwarder.
- `accumulate` calls the chunk stage with a chunk and the forwarder's value, and pushes into the accumulators.
- `finalize` calls the finalize stage.

The accumulator classes, the validation of key sets, `on_finalize`, `clear`, and `visualize` stay.
`visualize` classifies nodes from `Stage.frontier` and `Stage.dynamic`; whatever else it needs from sciline is added with the rewrite.
`allow_bypass` becomes a derived property: a dynamic key that is also an input of the finalize stage.
The module shrinks to its policy, which values are transient and which are held, as scipp/ess#732 proposed.

`test_stream_of_chunks_with_context_held_between_changes` in `tests/stage_test.py` runs this shape, including a context update that keeps the accumulator.
The rewrite of the real class against its existing tests has not been done.

### 6.7 essapps

essapps is the planned service for automatic, batch, and interactive data reduction.
Its design is on the `architecture-sketch` branch (`docs/developer/architecture.md`), and `docs/developer/stages.md` there reads it against this proposal.
In that design a workflow is described by a *spec* that declares its parameters and outputs, every run produces a stored *record*, and a *binding* connects a spec to the sciline workflow.
Three of its mechanisms are stages and aggregations:

- **Warm workflow (D8).**
  For interactive work the spec declares cheap parameters, and a rerun after changing one of them should reuse everything upstream.
  This is `Stage(pipeline, inputs=cheap_parameters, outputs=targets)`, with the cache being the frontier of the stage.
  The cheap parameters stay declared, because they decide what the user interface offers as a slider.
  A cheap parameter that the targets do not need is rejected when the stage is built.
- **Declared additive combine (D15).**
  A workflow can declare a contribution at its accumulation keys, and the service then runs contribute, combine, and finalize as separate steps, which are the three entry points of `Aggregation`.
  Where the contributions are kept depends on how the service runs:
  - In batch and automatic reduction, each new member of a series leads to a combine request in a short-lived process: `agg.finalize(agg.combine([previous, new]))`, with nothing held between requests.
  - In an interactive session, the wrapper holds the contributions by member label; removing a member is a combine over the remaining ones.
  - A long-running process (called a *fold* in essapps) holds accumulators from `agg.accumulators()`, pushes each new contribution, and writes `value` as a record every n contributions.

  The spec still declares which parameters finalize reads (D13), because the service validates a combine request without importing workflow code.
  The binding derives the actual split from `contribute_stage.keys` and `finalize_stage.keys` and rejects a spec whose declaration disagrees with the graph.
  Structure inside one record, such as angle groups in a Bifrost run or chunks of an NMX file, is an inner aggregation (section 6.3).
- **Interactive applications (phase 3).**
  essapps compares three models for interactive work: a session that holds state, an application that holds state itself, and a stateless service that splits a workflow into two specs with the intermediate value stored as a record.
  In all three, the objects are the same: a stage, and a connector after it.
  The models differ only in where these objects live and whether the value between the stages is written as a record.
  Choosing a model is therefore a question of placement, not of a new mechanism.

## 7. Changes to `Pipeline`

The breaking release combines this change with the PEP 695 generics prototype from the `235-pep695-single-model-prototype` branch (scipp/sciline#237).
This document supersedes that branch's design document where map/reduce is concerned: its open questions that came from map/reduce disappear.

**Removed:** `map`, `reduce`, `index_names`, `indices`, `get_mapped_node_names`, `compute_mapped`, computing a `pandas.Series` of keys, `visualize(compact=)`, `constraints=`, and cyclebane.
The graph becomes a plain `networkx.DiGraph`, and grafting with `__getitem__`/`__setitem__` needs about a hundred lines of networkx.

**Added:** `provide(key, callable)`, which registers a provider for a key known only at run time, so that `StreamProcessor` no longer needs to patch annotations (scipp/sciline#241).

**Kept, with the same interface:** construction from providers and `params`, `__setitem__` (values and grafted sub-pipelines), `__getitem__`, `insert`, `copy`, `get`, `compute`, `visualize`, `bind_and_call`, `output_keys`, and `underlying_graph` as a `networkx.DiGraph` with `value` and `provider` node attributes (read by `ess.reduce.workflow`).
Scheduler, task graph, reporter, handlers, and serialization are reused unchanged.
The generics come from the branch unchanged: rules are matched structurally, not by `TypeVar` identity, and only backward chaining is used.
`Scope` can then be deprecated (scipp/sciline#233).

**Required by `Stage`:** building the task graph for targets when some keys have no value, which `HandleAsComputeTimeException` does today (section 3, implementation notes).
The generics branch must keep this.

With demand-driven generics, `underlying_graph` and `output_keys()` only contain what has been requested.
`Stage` is not affected, because it builds the concrete task graph of its outputs and never looks at the sinks of the pipeline.
Callers that ask whether a stage uses a key must therefore test against `stage.keys`, not against the pipeline's graph.

**Tests and docs:** of the current 238 tests, 16 use map/reduce and are removed; the map-related tests of the generics branch are trimmed; the rest are ported.
The parameter-tables guide is replaced by the guide on stages and aggregations, and the generic-providers guide loses `constraints=`.

## 8. Design choices

### 8.1 Connectors as separate objects, not tiers inside `Stage`

The alternative was a `Stage` with several tiers of inputs and a policy per tier for what to hold, deriving all boundaries itself.
It carries the same information, but hides it inside one class and puts policy into sciline.
With separate connectors, every held value is an object the driver can inspect, clear, serialize, or send to another process.
This is the explicit lifetime that scipp/sciline#241 asks for.
`Stage` holds only its frontier, which is a forwarder from a stage without inputs, kept inside because it is the common case.

### 8.2 No generic network object (deferred)

A general object could hold stages and connectors and route pushes through them.
The real uses disagree on exactly this routing: a chunk runs immediately because it cannot be kept, a context update runs immediately but must not touch the accumulators, a table aggregation can run lazily, `clear` discards accumulators but keeps the context, and rolling windows and `on_finalize` add their own rules.
A generic object would either expose parameters for all of this or hide one choice.
It would also be a graph of stages with its own scheduler, which is nested workflows one level up (section 8.6).
Each real use is a loop of under twenty lines over plain objects, and that loop is where the policy belongs.

If package objects turn out to repeat the same code (for example esssans, essreflectometry, and the essapps wrapper), that repetition is the basis for a generalization of the package objects, not of `Aggregation`.
If essapps phase 3 needs connectors placed on process boundaries with pushes routed across them, that is a placement layer over the same stages and connectors, and should be designed with that case at hand.

### 8.3 No builder that derives stage boundaries (deferred)

A function could derive the boundaries from ordered groups of inputs.
Only one caller needs a derived boundary today, the context frontier of `StreamProcessor`, and it takes two lines (section 6.6).
The accumulation keys of an aggregation are declared, not derived.
The function should be extracted when a second caller appears.

### 8.4 An aggregation without state

An earlier draft put member tables with groups, the held contributions (matched to rows by comparing values), a held per-member frontier, and invalidation rules in `__setitem__` into the aggregation.
It rebuilt the map/reduced pipeline in another form, and it was the hardest part of the design to reason about: to predict the cost of a parameter change, one had to know which of five held things read the key.

The per-member frontier was meant to avoid reloading runs when a parameter changes.
On LoKI it gave no measurable benefit: the wavelength conversion reads the parameter `WavelengthBins`, so the per-member frontier lies upstream of the conversion and holding it saves little.
Holding more would require a second tier of parameters declared as rarely changing, which is what `StreamProcessor` calls context.
The draft was stripped down to the current `Aggregation`, and its rules moved into the package object (section 6.1).

### 8.5 No bridge back into a pipeline

Another draft had `as_pipeline()`, which added providers for the accumulation keys so that the `with_*` helpers could keep returning a pipeline.
It was dropped for three reasons.
It turned out to be the only way two aggregations could be composed, so it was the actual mechanism, not a convenience.
It fixed the member table when the pipeline was built.
And it hid a loop inside a provider, with no per-member errors, progress, or visualization.
The trap to avoid remains: a hand-written provider that runs a pipeline.

### 8.6 Not nested workflows

Nested workflows, sub-workflows with their own parameters inside a node of an outer workflow, were considered and rejected earlier.
The problems were plumbing, since parameters had to be passed across boundaries, and opacity, since a boundary hid what was inside.

Here the author writes one flat graph.
Stages are cut from it by the caller, from the keys the caller names, at the time they are used.
Every parameter is set on the flat pipeline and reaches every stage, and nothing is added to the author's graph.
Composition across stages is ordinary Python with connectors in between, so no graph contains another graph.

### 8.7 Names

- **Aggregation** and **contribute/combine/finalize**, as in Spark, Flink, Beam, and pandas.
  `fold` was avoided because it means reshaping in scipp.
- **Accumulator** for the stateful object, as in Beam, Flink, and Spark.
- **Accumulation key**, matching the essapps term (formerly "accumulation point").
- `Forwarder` is the working name in ess.reduce; the final name is ess.reduce's decision.

## 9. Validation

### Unit tests

The prototype is in `src/sciline/stage.py` and `src/sciline/aggregation.py`, tested in `tests/stage_test.py` and `tests/aggregation_test.py`.
It runs on sciline `main`.
On the generics branch all tests also pass except the two that aggregate over a generic key; these fail in a check inside `map` and would pass once `map` is removed.

The tests cover:

- **Stage:** static part computed once and dynamic part per call; an intermediate input cuts off its ancestors; inputs the outputs do not need are rejected; pass-through of an output that is an input; snapshot behaviour; `warm` computes shared work once and skips warm stages; concurrent calls compute the static part once; an expensive load before a cheap parameter (the warm-workflow shape).
- **Accumulators:** push order, the first push as result, reading without pushes.
- **Aggregation:** equal to a manual loop; two accumulation keys with different accumulators; a two-column member table; a key that does not depend on the members; members the accumulation keys do not need are rejected; the three steps called separately, with chained combining; no outputs; grouping with pandas; snapshot of parameters and graph; `compute` pushes each contribution before making the next; generic keys with `TypeVar` providers; `compute_members`.
- **Composition:** a caller that keeps contributions by label and rebuilds on a parameter change only when the contribute stage reads the key; banks over runs; two aggregations sharing a finalize stage; the `StreamProcessor` shape with a context update.

### LoKI multi-run reduction

`loki_validation.py`, next to this document, runs the esssans multi-run test workflow with one mask file, two sample runs, and two background runs.

- **Reference:** `with_pixel_mask_filenames`, `with_sample_runs`, and `with_background_runs`, using map/reduce.
- **Prototype:** `SansReduction` (section 6.1) with two aggregations on the flat pipeline, and the masks as a list parameter (section 6.4).

Results:

- `BackgroundSubtractedIofQ` and `BackgroundSubtractedIofQxy` are identical to the reference (`assert_identical`).
- Per-member `NormalizedQ` equals both a single-run computation and `compute_mapped`.
- Calling contribute, combine, and finalize separately on plain aggregations gives the same result as `SansReduction.compute`.
- Provider call counts equal the reference, including a single read of the mask file.
  This requires `warm` over all three stages; without it, each stage reads the file.
- Wall time with the naive scheduler: 6.9 s for the prototype, 7.2 s for the reference.
  With sciline's default dask scheduler the reference is about 1.7 s faster, because the single graph computes the two sample runs in parallel threads.
  In the prototype this parallelism would be up to the caller (by mapping `contribute` over the runs) and is not used.
- Adding a second sample run after computing with one costs one contribution: one more `apply_pixel_masks` call and no second read of the mask file.
- Changing `QBins` drops the contributions and recomputes them, as the reference does.

Not validated: the rewrite of `StreamProcessor` against its real tests.

## 10. Migration

### What changes for each project

- **sciline:** adds `Stage`, `warm`, `Accumulator`, `Buffered`, `Reduced`, `Aggregation`, `compute_members`, and `Pipeline.provide`, and later removes what section 7 lists.
  `Aggregation` belongs in sciline because it is mechanism, not policy: it holds no contributions, no member table, and no invalidation rules, and users outside ESS need a documented replacement for `map(...).reduce(...)`.
  It ships without an "experimental" label; the staged rollout below is its trial period.
  `Stage` also needs a `reporter` argument, so that progress reaches the ESS widgets.
- **ess.reduce:** the accumulators satisfy `sciline.Accumulator` once `maybe_hist` moves out of their base class, and a `Forwarder` is added.
  `StreamProcessor` is rewritten on `Stage` against its existing tests (section 6.6).
  `parameter_mappers`, which maps a list-valued parameter to a `with_*` helper that returns a pipeline, is replaced by a common interface of the package objects (section 11).
  `get_parameters` is unaffected, because it works on the flat pipeline.
- **Reduction packages** (esssans, essreflectometry, essspectroscopy, essdiffraction, essnmx): the `with_*` helpers that fold are replaced by package objects on which users set members and parameters and compute (section 6.1).
  Notebooks and tests change accordingly, and `visualize(compact=)` is dropped from notebooks.
  Specific points:
  - The pixel-mask folds in esssans and essdiffraction become a list parameter with providers (section 6.4).
  - `with_banks` in esssans, which maps without reducing, becomes a loop or `compute_members`.
  - essreflectometry drops the `try/except` around each reduce, since `accumulation_keys` reports keys that are not accumulated, and its `BatchProcessor` loses the fallback for mapped pipelines.
  - In bifrost, `NeXusData` depends on the run, so a bank fold inside a multi-run reduction sits inside per-run work, as with the esssans masks.
    The package object then either holds a bank aggregation per run, or the banks become a list parameter with a provider that loops.
- **esslivedata:** the bifrost bank fold becomes an `Aggregation` computed before the `StreamProcessor` is built (section 6.5).
- **essapps:** the design text is updated as listed at the end of `stages.md`.
  The fake workflow with two accumulation keys in the D3/D6 spike becomes an `Aggregation`.

### Findings of the survey that affect the migration

From the survey of 2026-09-11 (section 1):

- essreduce is the only package in the scipp/ess monorepo that depends on sciline directly (`sciline>=25.11.0`).
  The breaking release is adopted by raising that minimum.
- In ess.reduce, `assign_parameter_values` goes together with `parameter_mappers`; both are used only by `WorkflowWidget`.
  The polarization notebook `zoom.ipynb` in ess.reduce uses `get_mapped_node_names` and `with_sample_runs` and migrates with esssans.
- essdiffraction's `with_pixel_mask_filenames` has no test coverage: all 22 call sites in tests pass an empty list.
  Its migration should add a test with a mask file, and the workaround for empty lists in cyclebane goes.
- essreflectometry (offspec, amor) uses `constraints=`, which stays until the breaking release.
- essnmx notebooks import `cyclebane.graph.NodeName` and `IndexValues` directly to name mapped nodes.
- essimaging uses map/reduce only through `visualize(compact=)` in two notebooks.
- esslivedata pins `cyclebane>=26.9.0` directly, for a leak with self-referential graphs; the pin goes when sciline drops cyclebane.
- Tracking: one issue in scipp/sciline for the sciline releases, one in scipp/ess for ess.reduce and the reduction packages.

### Order

1. sciline releases the additive part in a minor version, with the user guide on stages and aggregations.
2. The packages migrate one at a time, each in its own pull request, while map/reduce still exists.
   esssans goes first, because it is the validated case and sets the pattern for package objects and for the interface that replaces `parameter_mappers`.
   The other packages depend only on the additive release and can then migrate in parallel.
   The `StreamProcessor` rewrite follows independently, but needs `Pipeline.provide` and the `reporter` argument of `Stage` in a sciline release first.
   esslivedata follows the `StreamProcessor` rewrite through the essreduce version bump.
3. The last minor release of sciline deprecates `map`, `reduce`, `compute_mapped`, `get_mapped_node_names`, `constraints=`, and `visualize(compact=)`, once esslivedata has migrated.
4. Once no ESS package or esslivedata uses map/reduce, a major release of sciline removes it together with cyclebane, and the PEP 695 generics land.

## 11. Open questions

- **One package object per package, or one generic object?**
  The widgets need a common interface (set a parameter, set the members, compute).
  Either each package object implements it, or one generic object is built from a registry that maps member keys to accumulation keys.
  This is decided when the second package migrates, when it is clear how much code the package objects share.
- **Static work across processes.**
  Stages in one process share their static work through `warm`.
  A contribute call in a short-lived process recomputes it.
  This is the cost of essapps phase 3 that its stateless-service notes already measure, not a new cost.
- **`sciline.v2` or a major release?**
  A `v2` namespace would let esslivedata and external users keep the old `Pipeline` next to the new one.
  But it bundles two independent changes (generics and map/reduce) under one name, invites mixing old and new pipelines in one process with obscure failures, and guarantees a second rename later.
  Recommendation: one major release, preceded by the additive minor release and the package migrations; esslivedata pins the previous version until it follows; the last minor release deprecates `map`, `reduce`, and `constraints=`.
