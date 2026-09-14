# Map and reduce outside the graph

Proposal, with a prototype. Companion to the demand-driven generics design document on the `235-pep695-single-model-prototype` branch (scipp/sciline#237), which it supersedes in one respect: the open questions there that came from `map`/`reduce` disappear if `map`/`reduce` do.

## Summary

Sciline's `map`/`reduce` put a loop over members and a combine step inside the graph.
That has been tried twice (parameter tables in 23.08, cyclebane in 24.06), is the source of every breaking change and every remaining open question in the PEP 695 work, and still cannot express what `ess.reduce.streaming.StreamProcessor` and the essapps architecture need: run one part of a graph repeatedly with some values supplied per call, combine the results outside, run the rest once.

The proposal is to make that operation the primitive and compose everything else from it outside the graph:

- `Pipeline` (called v2 below, whether it ships as a namespace or a major release; see Migration): the flat graph from type hints, with PEP 695 generics, without `map`, `reduce`, `groupby`, `constraints`, or cyclebane.
- `Stage`, in sciline: the part of a pipeline from a set of input keys to a set of output keys, with everything that does not depend on the inputs computed once and the inputs supplied per call. This is what `StreamProcessor` builds by hand today, what scipp/sciline#241 asks for, and what the essapps warm workflow (D8) and split workflows (phase 3) are.
- `Accumulator`, in sciline: the structural protocol for what sits between stages, `push` a value and read the combined `value`. `Buffered(func)` is a factory for an accumulator that holds all pushed values and applies an n-ary function on `value`; `Reduced(func)` is a factory for one that holds only a running result of an associative binary function. The accumulators of `ess.reduce.streaming` satisfy the protocol. `Forwarder`, which holds the latest value pushed, is the connector for a context between stages and stays in ess.reduce.
- `Aggregation`, in sciline: the table-fold shape as two stages of one pipeline, contribute and finalize, with an accumulator per accumulation key between them and the three entry points exposed. It holds nothing but its stages; parameters are set on the pipeline, and whoever loops over members owns the contributions.
- `StreamProcessor`, in ess.reduce, as a loop over stages and connectors carrying only its policy.

There is no drop-in replacement for the pipeline that `with_sample_runs` and its siblings return today.
Each package returns its own object, about thirty lines over `Aggregation`, that holds the pipeline, the aggregations, the contributions, and the "set runs, set parameters, compute" experience.

A prototype on sciline `main` passes 27 tests covering the use-case shapes found in the ESS packages and reproduces the LoKI multi-run reduction identically against the existing `with_sample_runs`, with the same provider call counts.

## Problem

### In sciline

`map` relabels every reachable node at call time, before the targets are known.
The demand-driven generics work paid for that twice: first with forward chaining and its seeding, which the Q1 spike removed by deferring the labeling into cyclebane (scipp/cyclebane#32, unreleased, which is why the branch's CI is red), and then with what the deferral cost: the `reduce(key=)` sink guard, `_consumed_by_template`, the mapped-root case in `_satisfied`, the node-name uniqueness break that forbids the `pipeline[C] = pipeline[C].map(...).reduce(...)` idiom every ESS package uses, and the `get_mapped_node_names` rework.
Those are the breaking changes of scipp/sciline#237, and every one of them is map/reduce-attributable.
Users cannot write a mapped key; they need `get_mapped_node_names` and `compute_mapped`, which depend on pandas and on cyclebane's node-name classes.
`groupby` was never exposed by sciline, and `index_names`/`indices` are unused downstream.

### In ess.reduce

`ess.reduce.streaming` is 1080 lines, about half of it `StreamProcessor`.
Its core is a graph partition: the ancestors of the targets, the descendants of the dynamic keys, and the frontier between them, computed once.
Sciline gives it no way to express that, so it assigns `None` to keys to prune branches, grafts subgraphs through `__setitem__`, and after scipp/ess#732 feeds values through providers whose `__annotations__` are patched at runtime.
scipp/sciline#241 records the three requirements the mechanism must meet: no graph rebuild per value, explicit lifetime of supplied values, and culling of what a supplied key's ancestors would otherwise compute.

### In essapps

Three parts of the architecture are the same partition:

- D8, the warm workflow: cache the nodes just upstream of the cheap parameters, rerun what lies downstream.
- D15, the declared additive combine: the graph up to the accumulation keys is contribute, the graph from the keys to the targets is finalize, combine is outside both.
- Phase 3, the split workflow: the same partition, but the value at the boundary crosses a spec boundary as a stored stage output.

The architecture notes that `StreamProcessor` "is the additive structure made explicit" and that the wrapper "gets the three from the accumulation keys".
It does not yet have a name for the thing that does the partitioning.

## What map/reduce is used for

Every use in `/workspace/ess` and esslivedata, by shape:

| Shape | Sites | What is combined |
|---|---|---|
| Table fold: map over a table, reduce at one or more keys, result feeds further providers | esssans runs (four map/reduce pairs, one per reduced key, `merge_contributions`), essreflectometry runs (up to 7 reduced keys, concat and `_any_value`), isissans zoom monitors (concat along a new dim, assert-unique position), bifrost triplets (three folds across two workflow builders, `merge_triplets` and `concat_event_lists`), NMX panels and MTZ files, DREAM detectors with a two-column table | events by concat, dense by sum, metadata by pick-one, `DataGroup` by key |
| Fold whose reduce sits inside another fold's per-member work | pixel masks in esssans: `DetectorMasks` reads the detector IDs of the sample run. DREAM's mask fold is static (its reader takes only the filename) and no test exercises it | dicts by union |
| Per-member results, no reduce | esssans `with_banks`, LoKI notebook, `BatchProcessor`, bifrost test | none; read with `compute_mapped` |
| Several folds on one pipeline, one final stage | sample runs and background runs | as above |
| Fold in the static part of a `StreamProcessor` | esslivedata bifrost `EmptyDetector` over banks | `_combine_banks` |

Two things stand out.
Reduced nodes are grafted back onto the same pipeline (`workflow[K] = workflow[K].map(df).reduce(func=f)`), so the caller sees an ordinary pipeline afterwards, and `parameter_mappers` in `ess.reduce` depends on that; the proposal gives that up, see the migration.
And essreflectometry wraps each reduce in `try/except` because a reduced key may not depend on the mapped key at all.

## Proposal

### 1. `sciline.v2.Pipeline`

The pipeline from the `235-pep695-single-model-prototype` branch, minus map/reduce, minus cyclebane.

Kept, unchanged in interface: construction with providers and `params`, `__setitem__` (values and grafted sub-pipelines), `__getitem__` (ancestor subgraph), `insert`, `copy`, `get`, `compute`, `visualize`, `bind_and_call`, `output_keys`, `underlying_graph` as a `networkx.DiGraph` with `value` and `provider` node attributes, since `ess.reduce.workflow` reads those.
`scheduler`, `task_graph`, `reporter`, `handler`, `serialize`, `_provider`, `display` are reused as they are.
Generics: `_unification.py` and the template registry from the branch, verbatim; backward chaining only.
PEP 695 syntax works because rules are matched structurally, not by `TypeVar` identity, and `Scope` can be deprecated (scipp/sciline#233).

Dropped: `map`, `reduce`, `index_names`, `indices`, `get_mapped_node_names`, `compute_mapped`, `compute` of a `pandas.Series`, `visualize(compact=)`, `constraints=`, and with them the reduce guards, the dependency on an unreleased cyclebane, and cyclebane itself.
The graph is a plain `networkx.DiGraph`; `__getitem__`/`__setitem__` grafting is about a hundred lines of networkx.

Added: `provide(key, callable)`, the entry point scipp/sciline#241 asks for, so that a provider for a key known only at runtime does not need patched annotations; and building the task graph for targets without values for some keys, which `HandleAsComputeTimeException` already does and `Stage` needs.
One consequence of demand-driven generics to keep in mind: `underlying_graph` and `output_keys()` see only what has been demanded.
`Stage` is built from the concrete task graph of its outputs and never derives a default from the pipeline's sinks, so this does not affect it; `stage.keys` is the concrete key set, which is what callers test membership against.

Tests: 238 today; 16 use map/reduce and go, the PEP 695 branch's map-related tests are trimmed, the rest port.
Docs: the parameter-tables guide is replaced by a guide on stages and aggregations; the generic-providers guide loses `constraints=`.

### 2. `Stage`

```python
stage = Stage(pipeline, outputs=(Numerator, Denominator), inputs=(Filename,))
stage.frontier          # the static keys the dynamic part reads, computed once
values = stage({Filename: 'run1.nxs'})   # -> {Numerator: ..., Denominator: ...}
warm(stage_a, stage_b)  # static parts of several stages of one pipeline in one run
```

Semantics:

- The graph is the ancestors of `outputs`.
  The dynamic part is `inputs` and their descendants; the static part is the rest.
  An input may be a parameter or an intermediate node; either way its own providers and ancestors are cut off.
  An output that is also an input is passed through, so that a stage from a key to itself is the identity; an aggregation whose members are values at its own accumulation key needs this.
- The frontier, the static keys read by dynamic nodes plus static outputs, is computed on first use and held.
  Nothing else of the static part is kept.
  `warm` computes the static parts of several stages together, so that what they share, a file every stage reads, is computed once and released; without it each stage would read it.
- A call supplies exactly the inputs, computes only the dynamic part, and releases the supplied and computed values when it returns.
  This meets the three requirements of scipp/sciline#241 by construction: no rebuild, call-scoped lifetime, culling.
- A stage is a snapshot: the pipeline it was built from is not modified, and changing that pipeline afterwards does not affect the stage.
  To change a fixed parameter, build a new stage.
- `dynamic`, `dynamic_outputs`, and `keys` expose the partition: which nodes and outputs depend on the inputs, and which keys the stage reads at all.
  Layered partitions, such as `StreamProcessor`'s static, context-dependent, and chunk-dependent sets, and the member frontier of an aggregation, are built from these, see below.

What it replaces: `_build_streaming_workflow`, `_FedWorkflow`, `_find_descendants`/`_find_parents`, and the pruning-by-assignment hack in `StreamProcessor`; the sciline wrapper of D8, whose cache is the frontier of `Stage(inputs=cheap_parameters)`; and each half of a split workflow.

It belongs in sciline because it needs the concrete graph and `Provider` to do without private access, which is the argument scipp/sciline#241 makes.
`Aggregation` needs nothing beyond `Stage` and the `Accumulator` protocol, but belongs in sciline too, because sciline documents map/reduce for all users and `Aggregation` is its replacement; see Migration.

### 3. Composition: stages and connectors

Every use above is a set of stages of one flat pipeline with something between them, and the something is where the state and the policy live.
Between two stages sits a connector, an object with `push` and `value`:

- an accumulator, where per-member values are combined.
  `Accumulator` is a structural protocol in sciline, `push(value)` and a `value` property, because `Aggregation` consumes it.
  `Buffered(func)`, in sciline, is a factory for an accumulator that holds all pushed values and applies an n-ary function on `value`.
  `Reduced(func)`, in sciline, is a factory for an accumulator that holds only a running result, applying an associative binary function to it and each pushed value; the function must not modify its arguments, since pushed values belong to the caller.
  The accumulators of `ess.reduce.streaming` satisfy the protocol as they are once `maybe_hist` moves out of their base class; `clear` stays an ess.reduce convention for accumulators that are reused between finalizes.
- a forwarder, where a value is held until it is replaced: `StreamProcessor`'s context, the frontier of a warm workflow, a stage output crossing a process boundary in essapps phase 3.
  `Forwarder` stays in ess.reduce; nothing in sciline consumes one.
- a dict by member label, where per-member contributions are held by whoever loops over members.

Whether a combine buffers or runs incrementally is the accumulator's choice, not the aggregation's.
For concat-like combines the two cost about the same, twice the total, and `Buffered` is right.
For a sum over large dense arrays a running total holds one array instead of one per member, and an incremental accumulator such as `Reduced` is right.
An accumulator in the table case and in streaming is the same object: whether values arrive from one input over time or from many members is the driver's picture.
Lifetime (`clear`), identity (contributions by label), and order dependence are the driver's.

`Stage` itself holds nothing but its frontier, and even that is a forwarder from a stage with no inputs, kept inside because it is the common case.
Every held value is an object the driver can inspect, clear, serialize, or place on a process boundary, which is the explicit lifetime scipp/sciline#241 asks for.
A value leaves one driver and enters another as a parameter of the flat pipeline: an aggregation over banks whose result feeds a `StreamProcessor` is `pipeline[EmptyDetector] = banks.compute(table)[EmptyDetector]` before the processor is built.

The alternative considered was a `Stage` with tiers of inputs and a hold policy per tier, deriving all boundaries itself.
It is the same information, held inside one class instead of shown as objects, and it puts policy into sciline.
The connector form was chosen.

Two things were considered and deferred:

- A generic network object holding stages and connectors and scheduling pushes through them.
  The real shapes disagree on exactly that policy: a chunk runs eagerly because it cannot be held, a context update runs eagerly but must not touch the accumulators, a table aggregation may run lazily, `clear` drops accumulators but keeps context, rolling windows and `on_finalize` add their own rules.
  A generic object either parametrizes all of that or hides one choice, and it is the nested-workflow shape one level up: a graph of stages with its own scheduler.
  Each real shape is a loop of under twenty lines over plain objects, and the loop is where the policy belongs.
  If phase 3 needs connectors placed on process boundaries with pushes routed across them, that object is a placement and routing layer over the same stages and connectors, decided with that case in hand.
- A builder that derives the stage boundaries from the graph given ordered groups of inputs.
  One caller needs a derived boundary today, the context frontier of `StreamProcessor`, and it is two lines (the stream-shaped test in `tests/stage_test.py`).
  An aggregation's accumulation keys are a declaration, not a derivation.
  Extract the function when a second caller appears.

What is shared, then, is small: `Stage`, `warm`, `Accumulator`, `Buffered`, `Reduced`, and `Aggregation` in sciline; `Forwarder` and the existing accumulators in ess.reduce, satisfying the protocol without a base class; a visualize function over a list of stages and connectors, since `StreamProcessor`'s node classification is read off `Stage.frontier` and `dynamic`; and the drivers below.

### 4. `Aggregation`

```python
agg = Aggregation(
    pipeline,
    members=(Filename[SampleRun],),
    accumulators={CleanSummedQ[SampleRun, Numerator]: Buffered(merge_contributions), ...},
    outputs=(IofQ,),
)
agg.accumulation_keys           # the keys in accumulators that depend on the members
agg.contribute(row)             # one member -> {accumulation key: value}
agg.accumulators()              # fresh accumulators, one per accumulation key, for a caller that pushes
agg.combine([...])              # push each contribution into fresh accumulators, read them
agg.finalize(contribution)      # accumulation keys -> outputs
agg.compute(table)              # contribute per row, pushing each as it is made; combine; finalize
compute_members(pipeline, members=(Filename[SampleRun],), key=NormalizedQ[SampleRun, Numerator], table=table)
```

Semantics:

- An aggregation is two stages of one flat pipeline with an accumulator per accumulation key between them: contribute from the member keys to the accumulation keys, finalize from the accumulation keys to the outputs.
  Parameters are set on the pipeline before the aggregation is built, and the aggregation is a snapshot; a changed parameter means a new aggregation, which costs a graph walk and no computation until the aggregation is warmed.
- `accumulators` maps each accumulation key to a factory for its accumulator.
  `combine` and `compute` make fresh accumulators from the factories on every call, so nothing survives between calls.
  Factories rather than instances with a `clear`: clearing inside the aggregation would either forbid combining in batches over several calls or silently add to stale state, and holding instances would make `combine` non-reentrant; with factories, whoever calls `accumulators()` owns the instances and their lifetime.
  `Buffered(func)` wraps an n-ary function, today's `reduce(func=)` signature, which every combine in the ESS packages already has; every existing combine migrates through it.
  `Reduced(func)` wraps an associative binary function and keeps a running result; it does not update in place, so a sum briefly holds the old and new result and the pushed value.
  An accumulator that needs in-place updates is written by the workflow author, who owns the copy of the first value; the ess.reduce accumulators are such objects.
  A key in `accumulators` that does not depend on the members is not accumulated; finalize computes it from the fixed part of the graph.
  That removes essreflectometry's `try/except`; it also means a misplaced key is silently static, which `accumulation_keys` reports and a test should check.
- The aggregation holds nothing but its stages.
  Whoever loops over members owns the contributions, a dict by label, and decides what a parameter change keeps with one test: whether the key is in `contribute_stage.keys`.
  `compute(table)` is the loop for a caller that holds nothing: it contributes per row and pushes each contribution as it is made, so peak memory is the accumulators' choice, not the aggregation's.
  The table is a `Mapping[label, Mapping[Key, value]]`, which `df.to_dict('index')` produces; pandas stays out of the library and the label is always the caller's.
- `outputs` is optional.
  An aggregation used only for its contributions, one of several sharing a finalize stage, builds no finalize.
  Sample runs and background runs are two aggregations on one pipeline and one `Stage` from both sets of accumulation keys to the outputs, warmed together.
- `contribute`, `combine`, and `finalize` are the three entry points of essapps D15 and can run in three processes with the contribution, a dict at the accumulation keys, serialized between them.
- Hierarchy is composition of plain objects.
  Banks over runs: one aggregation per bank on a pipeline with the bank set, the per-bank results given as members of a second aggregation whose member key is its own accumulation key.
  Angle groups inside a run (essapps D15) are an inner aggregation whose finalize output is a member of the outer one.
  Nothing nests inside a graph.
- Not in `Aggregation`: `groupby`, which is a grouping of the table and an aggregation per group; parallelism over members, which is the caller's job, since `contribute` is a plain call the caller can map over rows with threads, processes, or dask delayed and push the results as they arrive, and the user guide shows an example; and held state of any kind.
  The single-scheduler run over all members that in-graph mapping gave is what this loses; it is also what made everything else hard.

**The package object.**
What esssans returns instead of a map/reduced pipeline is its own object, `SansReduction` in the validation, about thirty lines: the pipeline, one aggregation per run type, the finalize stage over both sets of accumulation keys, and the contributions by filename.
`set_runs` drops the contributions of runs no longer listed; `reduction[key] = value` sets the parameter on the pipeline, rebuilds the aggregations whose contribute stage reads the key and drops their contributions, and rebuilds finalize if it reads the key; `compute` warms, contributes what is missing, combines, finalizes.
That is the only place the "set runs, set parameters, compute" experience lives and the only place that decides what a change keeps, and it is visible in thirty lines.
An earlier draft put groups, held contributions, a held member frontier, and the invalidation rules into the aggregation itself.
It reproduced the map/reduced pipeline, and it was the part of the design that was hard to reason about: to predict what a parameter change cost one had to know which of five held things read the key.
If essreflectometry and the essapps wrapper turn out to write the same thirty lines, that repetition is the case for the network object deferred above, as a generalisation of the package objects, not of `Aggregation`.

A fold whose reduce sits inside another fold's per-member work is not an aggregation.
esssans's pixel masks are the case: `DetectorMasks` reads the detector IDs of the sample run, so under `map`/`reduce` the mask fold is evaluated per sample run by the graph.
Outside the graph that is one list parameter, `PixelMaskFilenames`, and two providers: one reads the files, static and shared; one builds the masks per run.
Aggregations are for members that are expensive or whose per-member values users want; a handful of small files combined by union is neither.

### 5. `StreamProcessor` on `Stage`

`StreamProcessor` is not an `Aggregation`: its members are not recomputable, its accumulators are reused between finalizes and may be non-associative (rolling windows), and a context update must not invalidate what was accumulated.
It is three stages and two kinds of connector, which is also how it is structured today, with the context frontier derived from the partition rather than found by hand:

```python
per_chunk = Stage(pipeline, outputs=accumulator_keys, inputs=dynamic_keys)
context_frontier = Stage(pipeline, outputs=per_chunk.frontier, inputs=context_keys).dynamic_outputs
context_stage = Stage(pipeline, outputs=context_frontier, inputs=context_keys)
chunk_stage = Stage(pipeline, outputs=accumulator_keys, inputs=dynamic_keys + context_frontier)
finalize_stage = Stage(pipeline, outputs=target_keys, inputs=accumulator_keys + context_targets)
context = Forwarder(); accumulators = {key: EternalAccumulator() for key in accumulator_keys}
```

The stream-shaped test in `tests/stage_test.py` runs this shape, including a context update with the accumulator kept; the rewrite of the real class against its 35 tests is the next step, not done here.

`set_context` calls the context stage and pushes into the forwarder; `accumulate` calls the chunk stage with a chunk and the forwarder's value and pushes into the accumulators; `finalize` calls the third stage.
The accumulator classes, the key-set validation, `on_finalize`, `clear`, and `visualize` stay, and the accumulator classes satisfy sciline's `Accumulator` protocol as they are.
Their base class histograms by default in `push` through `maybe_hist`; that moves to the reducing subclasses, so that the base class adds nothing to the protocol and a forwarder does not histogram.
`allow_bypass` becomes "a dynamic key that is also an input of the finalize stage".
The module shrinks to the policy, which is what scipp/ess#732 said should stay: which values are transient and which are held.

### 6. What this means for essapps

- D8: the wrapper is `Stage(inputs=cheap_parameters)`.
  The cheap parameters stay a declaration, since they decide what a UI offers as a slider; the cached nodes are derived from it, not declared.
- D15: contribute, combine, finalize are `Aggregation`'s three entry points; the contribution is the dict at the accumulation keys.
  The declaration of which parameters finalize reads is derived from the graph and can be removed from D13/D15.
  A chained series is an accumulator per accumulation key, from `agg.accumulators()`, into which essapps pushes each contribution as it arrives; the in-memory variant is the wrapper holding the contributions by member label.
- Phase 3: the session model's warm workflow, the checkpoint model's in-application workflow, and the split model's two stages are the same `Stage` objects; the models differ only in where the objects live and when a value at an accumulation key becomes a record, which is where a forwarder sits.
  The decision the stateless note defers is then about placement, not about a mechanism.
- Hierarchy inside one record (Bifrost angle groups, NMX chunks) is an inner aggregation whose finalize output is a member of the outer one, which is what D15 already says the callable does itself.

## Why this is not nested workflows

The nested-workflow idea that was rejected before sciline put a graph inside a node: sub-workflows with their own parameters, composed by an outer workflow.
Its problems were plumbing and opacity: parameters had to be passed through boundaries, and a boundary hid what was inside.

Here the author writes one flat graph, and stages are derived from it at the time they are used, from the keys the caller names.
Every parameter is set on the flat pipeline and reaches every stage.
No provider or key is added to the author's graph.
Composition across stages is function composition in Python with connectors between, so the graph never contains a graph, and what is held is visible as objects.

An earlier draft kept a bridge, `as_pipeline()`, that synthesized providers for the accumulation keys so that the `with_*` helpers could keep returning a pipeline.
It was dropped: it was the only way two aggregations could compose, so it was not a bridge but the mechanism; it forced the member table to be fixed at construction; and it reintroduced the opacity, a provider that runs a loop with no per-member errors, progress, or visualization.
The trap to keep out of stays: a provider written by hand that runs a pipeline.

## Evidence

Implementation: `src/sciline/stage.py` and `src/sciline/aggregation.py`, tests in `tests/stage_test.py` and `tests/aggregation_test.py`, on sciline `main` (the pep695 branch needs an unreleased cyclebane for generics; on it all tests pass except the two that aggregate over a generic key, which fail in the branch's mapped-root check inside `map` and so go away with it).
It reads `TaskGraph._graph` for the concrete graph and hardcodes the naive scheduler inside stages; both go away inside v2.

Covered:

- static part computed once, dynamic part per call; an intermediate input cuts off its ancestors; an input the outputs do not need is refused; an output that is an input is passed through; `warm` computes shared static work once;
- the warm workflow shape (cheap parameter after an expensive load);
- an aggregation with two accumulation keys and different accumulators equal to a manual loop; a two-column member table; an accumulation key independent of the members passed through; members the accumulation keys do not need refused; contribute, chained combine, finalize as separate calls; an aggregation without outputs; groupby as an aggregation per group of the table;
- the caller's pattern: contributions held by label, a new aggregation on a parameter change, contributions kept when the contribute stage does not read the key;
- banks over runs as one aggregation per bank feeding an outer aggregation; an aggregation over a generic key with `TypeVar`-instantiated providers; two aggregations sharing a finalize stage; the aggregation as a snapshot under later mutation of the caller's pipeline; per-member values of a key after the accumulation keys;
- the `StreamProcessor` shape: chunk stage, context in a forwarder, histogram in an accumulator, context update without clearing, finalize.

LoKI validation (`loki_validation.py` next to this document): the esssans multi-run test workflow, one mask file, two sample runs, two background runs.
Reference: `with_pixel_mask_filenames`, `with_sample_runs`, `with_background_runs` with `map`/`reduce`.
Prototype: `SansReduction`, the package object described above, over two aggregations on the flat pipeline, the masks as a list parameter.
`BackgroundSubtractedIofQ` and `BackgroundSubtractedIofQxy` are identical to the reference under `assert_identical`; per-member `NormalizedQ` equals both a single-run compute and `compute_mapped`; contribute, chained combine, finalize over bare aggregations equal the object's `compute`.
Provider call counts equal the reference's, including one read of the mask file, because `warm` computes the static parts of the two aggregations and the finalize stage together; without it the file is read once per stage.
Wall time is 6.9 s against 7.2 s under the same scheduler; under sciline's default dask scheduler the reference is about 1.7 s faster because the single graph runs the two sample runs in threads, and that parallelism is the caller's here, over `contribute`, not in the prototype.
Adding the second sample run after computing with one costs one contribution: one more `apply_pixel_masks` call and no second mask read.
Changing `QBins` drops the contributions and reruns everything from the loaded runs on, as the reference does.

A first draft held a per-member frontier, what the graph computes from a member alone, so that such a change would not reload.
On LoKI it bought nothing measurable: that frontier lies above the wavelength conversion, because the conversion reads `WavelengthBins`, a parameter.
Holding more would need a second tier of parameters declared as rarely changing, which is `StreamProcessor`'s context; it was removed from the aggregation.

Not shown by the prototype: the `StreamProcessor` rewrite against its real tests, and the user-guide example of parallel members over `contribute`.

## Migration

1. sciline: `Pipeline` without map/reduce, `Stage`, `warm`, `provide`, `Accumulator`, `Buffered`, `Reduced`, `Aggregation`, `compute_members`.
   `Aggregation` is in sciline because it is mechanism, not policy: it holds no contributions, no member table, and no invalidation rule, needs nothing beyond `Stage` and the `Accumulator` protocol, and is the documented replacement for `map(...).reduce(...)` that users outside ESS need.
   It ships without an experimental label; the staged rollout below, the additive minor release, then the esssans migration, then the breaking release, is the trial period.
   Whether this is `sciline.v2` or the next major release is a real choice.
   The namespace lets esslivedata and external users pin through the change and keeps v1 importable next to v2; it also bundles two independent changes under one name, invites mixing v1 and v2 pipelines in one process with obscure failures, and guarantees a second rename.
   Recommendation: one major release; the additive part (`Stage`, `warm`, `provide`, `Accumulator`, `Buffered`, `Reduced`, `Aggregation`, `compute_members`) ships in a minor release first so that the ESS packages migrate one at a time before removal; esslivedata pins the previous version until it follows; the last minor release before removal deprecates `map`, `reduce`, and `constraints=`.
2. ess.reduce: `Forwarder` and the accumulators in one module; the accumulators satisfy sciline's `Accumulator` protocol once `maybe_hist` moves out of their base class, and keep `clear` as their own convention; `StreamProcessor` rewritten on `Stage` against its existing tests.
   essreflectometry's `BatchProcessor` loses its mapped/unmapped fallback.
   `parameter_mappers`, which maps a list-valued parameter to a `with_*` helper returning a pipeline, goes.
   The UI needs one protocol across packages, set a parameter, set the members for a member key, compute; whether each package object implements it or one generic object is built from a registry of member key to accumulation keys is the "same thirty lines" question above, decided when the second package is migrated.
   `get_parameters` is unaffected because it runs on the flat pipeline.
3. esssans, essreflectometry, essspectroscopy, essnmx, essdiffraction: the `with_*` helpers that fold are replaced by a package object on which users set runs and parameters and compute; notebooks and tests change accordingly.
   `with_pixel_mask_filenames` in esssans becomes a `PixelMaskFilenames` list parameter with two providers; essdiffraction's, whose reduced key is static, becomes the same for uniformity and loses its empty-list workaround for cyclebane.
   `with_banks`, which maps without reducing, becomes a loop setting `NeXusDetectorName` on the package object or the pipeline, which is what its callers do with `compute_mapped` anyway.
4. esslivedata: the bifrost bank fold becomes an `Aggregation` computed before the `StreamProcessor` is built, its result set as a parameter of the processor's pipeline.
5. essapps: D8 and D15 wording as above; the D3/D6 spike's fake workflow with two accumulation points is an `Aggregation`.

## Rollout plan

Written 2026-09-11 from the site survey of the same date (every `map`, `reduce`, `compute_mapped`, `get_mapped_node_names`, `constraints=`, `compact=`, and `parameter_mappers` use in `/workspace/ess` and esslivedata) and the review of the prototype.
Each item is one pull request unless stated.
The rule for the order: everything additive lands and releases before anything breaks, and the breaking sciline release waits until no ESS package or esslivedata uses map/reduce.

### A. sciline, additive

1. ADR 0003 with `Stage`, `warm`, `Accumulator`, `Buffered`, `Reduced`, `Aggregation`, `compute_members`, their tests, and a user-guide page on stages and aggregations next to the parameter-tables page.
   Released as the next minor.
   Tracking issues: one in scipp/sciline for A and E, one in scipp/ess for B to D.
2. `provide(key, callable)` on `Pipeline` (scipp/sciline#241); a `reporter` argument on `Stage.__call__` and `warm` so that progress reaches the ESS widgets; whatever `StreamProcessor.visualize` needs to classify nodes from `Stage.frontier` and `Stage.dynamic`.
   Scoped when B2 starts, since B2 is the consumer.
3. Deprecation warnings on `map`, `reduce`, `compute_mapped`, `get_mapped_node_names`, `constraints=`, and `visualize(compact=)`, in the last minor before E.
   Lands when D1 is released.

### B. ess.reduce, additive

1. Accumulators: `maybe_hist` moves out of `Accumulator.push` into the subclasses that histogram; a `Forwarder` (name is ess.reduce's call); a test that every accumulator satisfies `sciline.Accumulator`.
   No behaviour change for `StreamProcessor` users.
2. `StreamProcessor` rewritten as a driver over three stages, a forwarder, and accumulators, keeping its interface, against `streaming_test.py`, `streaming_visualize_test.py`, and `accumulators_test.py`.
   `_FedWorkflow`, `_build_streaming_workflow`, `_find_descendants`, `_find_parents`, `_map_context_to_cached_nodes`, and the pruning by `None` go; `allow_bypass` becomes a derived property.
   Needs A2.
3. `assign_parameter_values` and `parameter_mappers` (`parameter.py:190`, `workflow.py:87-95`, consumed only by `WorkflowWidget.workflow_runner` in `ui.py:194`) are replaced by the package-object protocol.
   Lands with or after C1, which defines that protocol for esssans.
4. The polarization notebook `docs/user-guide/polarization/zoom.ipynb` uses `get_mapped_node_names` and `with_sample_runs`; it migrates with C1.

### C. Reduction packages, one PR each, in this order

1. **esssans**, the validated case and the one that sets the package-object pattern.
   `_set_runs`, `with_sample_runs`, `with_background_runs` (`workflow.py:104-108`, four map/reduce pairs per run type) become one aggregation per run type with `Buffered(merge_contributions)` and a shared finalize stage, held by the package object with the contributions by filename and a `clear`.
   `with_pixel_mask_filenames` (`:62-66`) becomes a `PixelMaskFilenames` list parameter and two providers, the mask built per run from that run's `DetectorIDs`.
   `with_banks` (`:92-94`, map without reduce) becomes `compute_members` or a loop.
   `ZoomTransmissionFractionWorkflow` (`isissans/zoom.py:158-165`) becomes an aggregation with `Buffered` over the concat and the unique-position check.
   The `parameter_mappers` registrations (`:144-150`) go.
   Notebooks: `loki-iofq`, `loki-direct-beam` (including the prose at cell `:537` about the `merge_contributions` node), `loki-reduction-ess` (cell 8, `compute_mapped` over banks), `isis/zoom`; `docs/api-reference/index.md`.
   Tests: `loki/iofq_test.py` (`compute_mapped` at `:213,239`), `isissans/zoom_reduction_test.py`, `i_of_q_test.py`.
2. **essdiffraction**, small.
   `with_pixel_mask_filenames` (`powder/masking.py:80-101`) becomes a list parameter and one provider; the empty-list workaround for cyclebane (`:94-97`) goes.
   All 22 test call sites pass `[]`, so the fold has no coverage today; add one test with a mask file.
   `dream-advanced-powder-reduction.ipynb` cells 30 and 39 (one- and two-column tables over detectors, `collect_detectors`) become an aggregation over a mapping table.
   The `parameter_mappers` registration in `dream/workflows.py:152` goes.
3. **essspectroscopy**, bifrost.
   Three folds over detector triplets: `RawDetector[SampleRun]` in `BifrostSimulationWorkflow` (`bifrost/workflow.py:112-116`), `EmptyDetector[SampleRun]` and `NeXusData[NXdetector, SampleRun]` in `BifrostWorkflow` (`:151-161`), with `merge_triplets` and `concat_event_lists`.
   Open design point to settle in this PR: `NeXusData` depends on the run, so the bank aggregation sits inside per-run work when runs are members, which is the esssans-mask shape; either the package object owns a per-run bank aggregation or a list parameter `DetectorNames` with a provider that loops.
   Test `bifrost/workflow_test.py:53-54` and notebook `bifrost-make-wavelength-lookup-table.ipynb:64` use `compute_mapped` and become `compute_members`.
4. **essreflectometry**.
   `with_filenames` (`reflectometry/workflow.py:63-86`) becomes an aggregation with up to seven accumulation keys, `Buffered` over `_concatenate_event_lists`, `_any_value`, and `_concatenate_lists`; the `try/except` around each reduce goes because `accumulation_keys` reports the static ones.
   `BatchProcessor.compute` (`tools.py:199-212`) loses the `compute_mapped` fallback for mapped-but-unreduced pipelines; `batch_processor` (`:561`) enters the aggregation for a list-valued `Filename[SampleRun]`.
   `gui.py:1037,1050` fold reference and sample runs.
   `constraints=` at `offspec/workflow.py:41` and `amor/__init__.py:90` stay until E.
   Notebooks `amor-reduction-advanced` (the `'611+612'` tuple entry) and `estia-advanced-mcstas-reduction`; tests `batch_processor_test.py:47,59`, `amor/pipeline_test.py`, `tools_test.py`.
5. **essnmx**, notebooks and one test fixture only.
   `mcstas_workflow.ipynb` folds panels over a scipp `Variable` and imports `cyclebane.graph.NodeName`/`IndexValues` directly to name mapped nodes; `scaling_workflow.ipynb` folds MTZ files with two accumulation keys, one of them the union `gemmi.SpaceGroup | None`; `tests/mcstas/workflow_test.py:38-43`.
6. **essimaging**: two notebooks use `visualize(compact=)`; drop the argument.
   Every package also drops `compact=` from its notebooks in its own PR.

### D. esslivedata

1. The bifrost bank fold (`config/instruments/bifrost/factories.py:375-379`) sits in the static part of the streaming pipeline; it becomes `Aggregation(...).compute(...)` with the result set as a parameter before the `StreamProcessor` is built.
   Consumes B2 through the essreduce bump.
   The direct `cyclebane>=26.9.0` pin (`pyproject.toml:36`, for a self-referential graph leak) goes when E removes cyclebane.

### E. sciline, breaking, next major

Remove `map`, `reduce`, `index_names`, `indices`, `get_mapped_node_names`, `compute_mapped`, `visualize(compact=)`, `constraints=`, and cyclebane; networkx becomes a direct dependency; the PEP 695 generics from `235-pep695-single-model-prototype` land without their map parts; `Scope` is deprecated (scipp/sciline#233); the parameter-tables guide is removed and the aggregations guide takes its place.
essreduce, the only direct sciline dependency in the monorepo (`sciline>=25.11.0`), raises its minimum; C3 and C4 drop their `constraints=` in the same essreduce bump.
Requires C1 to C5 and D1 released.

### F. essapps

Apply the edits listed at the end of `stages.md` on branch `architecture-sketch` (D8, D13, D14, D15, glossary), with `Fold` read as `Aggregation` and "accumulation point" as accumulation key; the D3/D6 spike's fake workflow with two accumulation points is an `Aggregation`.

### Scheduling

E waits for C and D; nothing forces it earlier.
The PEP 695 generics cannot land with map/reduce present, which is what this ADR is about, so they wait with E.
Parallelism between packages is possible after C1: C2 to C5 depend only on A1, and B2 only on A2.
The critical path is A1, C1, then the remaining packages, then E.

## Open questions

- **One package object or many.**
  See the migration; the answer decides whether the UI protocol is a base class or a convention.
- **Static work across processes.**
  Stages in one process share their static work through `warm`; a contribute in a throwaway process recomputes it.
  That is the phase 3 cost the essapps stateless note already measures, not a new one.

Settled 2026-09-11:

- **Combine protocol.**
  An accumulator per accumulation key, the `Accumulator` protocol with `push` and `value`, with `Buffered(func)` for n-ary functions and `Reduced(func)` for associative binary functions.
  An n-ary function forces buffering inside the aggregation, and anyone who cares about memory would write their own loop; with accumulators the buffer-versus-incremental choice sits on the object the author passes.
- **Member tables.**
  `Mapping[label, Mapping[Key, value]]` only; pandas stays out of the library and the label is always the caller's.
- **Parallel members.**
  The caller's job; `Aggregation` takes no executor.
  `contribute` is a plain call to map over rows with threads, processes, or dask delayed; the user guide shows an example.
- **Names.**
  `Fold` becomes `Aggregation`, "cut key" becomes "accumulation key", `at=` becomes `accumulators=`, `fold.cut` becomes `agg.accumulation_keys`; `Stage`, `warm`, members, contribution, `contribute`/`combine`/`finalize`, `contribute_stage`/`finalize_stage`, and `compute_members` are unchanged; `Forwarder` stays in ess.reduce, and its name is ess.reduce's call.
  `fold` already means reshaping in scipp, and contribute/combine/finalize is what Spark, Flink, Beam, and pandas call aggregation; in Beam, Flink, and Spark the stateful object is the accumulator.
- **Experimental label.**
  None; the staged rollout is the trial period.
