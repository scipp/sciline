# Map and reduce outside the graph

Proposal, with a prototype. Companion to the demand-driven generics design document on the `235-pep695-single-model-prototype` branch (scipp/sciline#237), which it supersedes in one respect: the open questions there that came from `map`/`reduce` disappear if `map`/`reduce` do.

## Summary

Sciline's `map`/`reduce` put a loop over members and a combine step inside the graph.
That has been tried twice (parameter tables in 23.08, cyclebane in 24.06), is the source of every breaking change and every remaining open question in the PEP 695 work, and still cannot express what `ess.reduce.streaming.StreamProcessor` and the essapps architecture need: run one part of a graph repeatedly with some values supplied per call, combine the results outside, run the rest once.

The proposal is to make that operation the primitive and compose everything else from it outside the graph:

- `sciline.v2.Pipeline`: the flat graph from type hints, with PEP 695 generics, without `map`, `reduce`, `groupby`, `constraints`, or cyclebane.
- `Stage`, in sciline: the part of a pipeline from a set of input keys to a set of output keys, with everything that does not depend on the inputs computed once and the inputs supplied per call. This is what `StreamProcessor` builds by hand today, what scipp/sciline#241 asks for, and what the essapps warm workflow (D8) and split workflows (phase 3) are.
- Connectors, in ess.reduce: objects between stages with `push`, `value`, and `clear`. The type of the connector says why the graph is cut there: a reducer where members are combined, a forwarder where a context is held between changes. The accumulators of `ess.reduce.streaming` are reducers in this sense.
- Drivers, in ess.reduce: `Fold` for the table-fold shape, with settable members and parameters and the three entry points contribute, combine, finalize; `StreamProcessor` for streams. Each is a loop over stages and connectors and carries only its policy.

There is no drop-in replacement for the pipeline that `with_sample_runs` and its siblings return today.
Those helpers become a dedicated object per package, built on `Fold`, on which users set runs and parameters and compute.

A prototype on sciline `main` passes 26 tests covering the use-case shapes found in the ESS packages and reproduces the LoKI multi-run reduction identically against the existing `with_sample_runs`, with the same provider call counts.

## Problem

### In sciline

`map` relabels every reachable node at call time, before the targets are known.
The demand-driven generics work paid for that twice: first with forward chaining and its seeding, which the Q1 spike removed by deferring the labeling into cyclebane (scipp/cyclebane#32, unreleased, which is why the branch's CI is red), and then with what the deferral cost: the `reduce(key=)` sink guard, `_consumed_by_template`, the mapped-root case in `_satisfied`, the node-name uniqueness break that forbids the documented `pipeline[C] = pipeline[C].map(...).reduce(...)` idiom, and the `get_mapped_node_names` rework.
Those are the breaking changes of scipp/sciline#237, and every one of them is map/reduce-attributable.
Users cannot write a mapped key; they need `get_mapped_node_names` and `compute_mapped`, which depend on pandas and reach into `_cbgraph`.
`groupby` was never exposed by sciline, and `index_names`/`indices` are unused downstream.

### In ess.reduce

`StreamProcessor` is 1081 lines.
Its core is a graph partition: the ancestors of the targets, the descendants of the dynamic keys, and the frontier between them, computed once.
Sciline gives it no way to express that, so it assigns `None` to keys to prune branches, grafts subgraphs through `__setitem__`, and after scipp/ess#732 feeds values through providers whose `__annotations__` are patched at runtime.
scipp/sciline#241 records the three requirements the mechanism must meet: no graph rebuild per value, explicit lifetime of supplied values, and culling of what a supplied key's ancestors would otherwise compute.

### In essapps

Three parts of the architecture are the same partition:

- D8, the warm workflow: cache the nodes just upstream of the cheap parameters, rerun what lies downstream.
- D15, the declared additive combine: the graph up to the accumulation keys is contribute, the graph from the keys to the targets is finalize, combine is outside both.
- Phase 3, the split workflow: the same cut, but the cut value crosses a spec boundary as a stored stage output.

The architecture notes that `StreamProcessor` "is the additive structure made explicit" and that the wrapper "gets the three from the accumulation keys".
It does not yet have a name for the thing that does the cutting.

## What map/reduce is used for

Every use in `/workspace/ess` and esslivedata, by shape:

| Shape | Sites | What is combined |
|---|---|---|
| Fold: map over a table, reduce at one or more keys, result feeds further providers | esssans runs (4 cut keys from one map, `merge_contributions`), essreflectometry runs (up to 7 cut keys, concat and `_any_value`), isissans zoom monitors (concat along a new dim, assert-unique position), bifrost triplets (two cut keys at different depths), NMX panels and MTZ files, DREAM detectors with a two-column table | events by concat, dense by sum, metadata by pick-one, `DataGroup` by key |
| Fold whose cut sits inside another fold's per-member work | pixel masks in esssans and esspowder: `DetectorMasks` reads the detector IDs of the sample run | dicts by union |
| Per-member results, no reduce | esssans `with_banks`, LoKI notebook, `BatchProcessor`, bifrost test | none; read with `compute_mapped` |
| Several folds on one pipeline, one final stage | sample runs and background runs | as above |
| Fold in the static part of a `StreamProcessor` | esslivedata bifrost `EmptyDetector` over banks | `_combine_banks` |

Two things stand out.
Reduced nodes are grafted back onto the same pipeline (`workflow[K] = workflow[K].map(df).reduce(func=f)`), so the caller sees an ordinary pipeline afterwards, and `parameter_mappers` in `ess.reduce` depends on that; the proposal gives that up, see the migration.
And essreflectometry wraps each reduce in `try/except` because a cut key may not depend on the mapped key at all.

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
One consequence of demand-driven generics to keep in mind: `underlying_graph` and `output_keys()` see only what has been demanded, so anything that derives a default from the pipeline's sinks, as `Fold` does below, must go through `output_keys()`, which on the branch includes the unconsumed rule patterns.

Tests: 335 today; 16 are map/reduce and go, 4 PEP 695 tests are trimmed, the rest port.
Docs: the parameter-tables guide is replaced by a guide on folds; the generic-providers guide loses `constraints=`.

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
  An input may be a parameter or an intermediate node; either way its own providers and ancestors are cut.
  An output that is also an input is passed through, so that a stage from a key to itself is the identity; a fold whose members are values at its own cut key needs this.
- The frontier, the static keys read by dynamic nodes plus static outputs, is computed on first use and held.
  Nothing else of the static part is kept.
  `warm` computes the static parts of several stages together, so that what they share, a file every stage reads, is computed once and released; without it each stage would read it.
- A call supplies exactly the inputs, computes only the dynamic part, and releases the supplied and computed values when it returns.
  This meets the three requirements of scipp/sciline#241 by construction: no rebuild, call-scoped lifetime, culling.
- A stage is a snapshot: the pipeline it was built from is not modified, and changing that pipeline afterwards does not affect the stage.
  To change a fixed parameter, build a new stage.
- `dynamic`, `dynamic_outputs`, and `keys` expose the partition: which nodes and outputs depend on the inputs, and which keys the stage reads at all.
  Layered partitions, such as `StreamProcessor`'s static, context-dependent, and chunk-dependent sets, and the member frontier of a fold, are built from these, see below.

What it replaces: `_build_streaming_workflow`, `_FedWorkflow`, `_find_descendants`/`_find_parents`, and the pruning-by-assignment hack in `StreamProcessor`; the sciline wrapper of D8, whose cache is the frontier of `Stage(inputs=cheap_parameters)`; and each half of a split workflow.

It belongs in sciline because it needs the concrete graph and `Provider` to do without private access, which is the argument scipp/sciline#241 makes.
It is the only thing here that does.

### 3. Composition: stages and connectors

Every use above is a set of stages of one flat pipeline with something between them, and the something is where the state and the policy live.
Between two stages sits a connector, an object with `push`, `value`, and `clear`:

- a reducer, where per-member values are combined: the accumulators of `ess.reduce.streaming`, or an n-ary function over held contributions;
- a forwarder, where a value is held until it is replaced: `StreamProcessor`'s context, the frontier of a warm workflow, a stage output crossing a process boundary in essapps phase 3;
- a store keyed by member label, where per-member values are held for reuse.

`Stage` itself holds nothing but its frontier, and even that is a forwarder from a stage with no inputs, kept inside because it is the common case.
Every held value is an object the driver can inspect, clear, serialize, or place on a process boundary, which is the explicit lifetime scipp/sciline#241 asks for.
A value leaves one driver and enters another as a parameter of the flat pipeline: a fold over banks whose result feeds a `StreamProcessor` is `pipeline[EmptyDetector] = fold.compute(EmptyDetector)` before the processor is built.

The alternative considered was a `Stage` with tiers of inputs and a hold policy per tier, deriving all boundaries itself.
It is the same information, held inside one class instead of shown as objects, and it puts policy into sciline.
The connector form was chosen.

Two things were considered and deferred:

- A generic network object holding stages and connectors and scheduling pushes through them.
  The real shapes disagree on exactly that policy: a chunk runs eagerly because it cannot be held, a context update runs eagerly but must not touch the accumulators, a table fold may run lazily, `clear` drops accumulators but keeps context, rolling windows and `on_finalize` add their own rules.
  A generic object either parametrizes all of that or hides one choice, and it is the nested-workflow shape one level up: a graph of stages with its own scheduler.
  Each real shape is a loop of under twenty lines over plain objects, and the loop is where the policy belongs.
  If phase 3 needs connectors placed on process boundaries with pushes routed across them, that object is a placement and routing layer over the same stages and connectors, decided with that case in hand.
- A builder that derives the stage boundaries from the graph given ordered groups of inputs.
  One caller needs a derived boundary today, the context frontier of `StreamProcessor`, and it is two lines (`stream_test.py`).
  A fold's cut is a declaration, not a derivation.
  Extract the function when a second caller appears.

What is shared, then, is small: `Stage` and `warm` in sciline; the connector protocol, a `Forwarder`, and the existing accumulators in ess.reduce; a visualize function over a list of stages and connectors, since `StreamProcessor`'s node classification is read off `Stage.frontier` and `dynamic`; and the drivers below.

### 4. `Fold`

```python
fold = Fold(
    pipeline,
    over={Filename[SampleRun]: cut_for(SampleRun),         # member key(s) -> {cut key: combine}
          Filename[BackgroundRun]: cut_for(BackgroundRun)},
    outputs=(BackgroundSubtractedIofQ,),
)
fold.set_members(pd.DataFrame({Filename[SampleRun]: files}).rename_axis('run'))
fold.set_members({Filename[BackgroundRun]: background_files})
fold[QBins] = bins                          # a parameter; what depends on it is rebuilt
fold.compute(BackgroundSubtractedIofQ)      # contribute what is not held, combine, finalize
fold.compute_members(NormalizedQ[SampleRun, Numerator])   # per-member values
fold.contribute(row); fold.combine([...]); fold.finalize([...])   # the three entry points
```

Semantics:

- `over` declares the structure: per group of member keys, the cut keys with an n-ary combine per key, today's `reduce(func=)` signature, which every combine in the ESS packages already has.
  A group is identified by its member keys, which are the columns of its member table.
  Sample runs and background runs are two groups of one fold with one finalize; a two-column table (DREAM) is one group with two member keys.
  A member key of one group is an ordinary parameter for the others: the LoKI background group reads `DetectorMasks` built from the detector IDs of whatever `Filename[SampleRun]` is set on the pipeline, as the reference does.
- A cut key that does not depend on the group's member keys is not folded; finalize computes it from the fixed part of the graph.
  That removes essreflectometry's `try/except`; it also means a misplaced cut key is silently static, which `cut` reports and a test should check.
- Per group, two stages of the flat pipeline: the member stage, from the member keys to the member frontier, the nodes that read nothing but the members; and the contribute stage, from the member frontier to the cut, whose static part is the outer frontier.
  One finalize stage from all cuts to the outputs.
  The split is derived with `Stage` alone: the member frontier is what the stage from the outer frontier to the cut reads that is not static.
  All static parts are warmed together.
- The fold holds contributions per member label, keyed by the table's row index.
  `set_members` keeps the held values of a member whose label and row are unchanged, so adding a run costs one contribution and replacing one costs one; `compute` contributes what is missing, combines each group with its functions in one n-ary call, and finalizes.
  `fold[key] = value` rebuilds only what reads the key: a parameter after the cut keeps every contribution; one before the cut drops them.
  With `keep_members=True` the member frontier, what the graph computes from a member alone, is held across such a change as well.
- `contribute`, `combine`, and `finalize` are the three entry points of essapps D15 and can run in three processes with the contribution, a dict at the cut keys, serialized between them.
  `contribute` takes a row, not a label, so a member need not be in any table.
- Hierarchy is composition of plain objects.
  Banks over runs: one fold over runs with `keep_members=True`, the bank set as a parameter per iteration, the per-bank results given as members of a second fold whose member key is its own cut key.
  Angle groups inside a run (essapps D15) are an inner fold whose finalize output is a member of the outer one.
  Nothing nests inside a graph.
- Not in `Fold`: `groupby`, which is a pandas `groupby` over the table and a fold per group; parallelism over members, which is a driver concern, threads or dask over `contribute`, not a property of the graph.
  The single-scheduler run over all members that in-graph mapping gave is what this loses; it is also what made everything else hard.

A fold whose cut sits inside another fold's per-member work is not a fold.
esssans's pixel masks are the case: `DetectorMasks` reads the detector IDs of the sample run, so under `map`/`reduce` the mask fold is evaluated per sample run by the graph.
Outside the graph that is one list parameter, `PixelMaskFilenames`, and two providers: one reads the files, static and shared; one builds the masks per run.
Folds are for members that are expensive or whose per-member values users want; a handful of small files combined by union is neither.

### 5. `StreamProcessor` on `Stage`

`StreamProcessor` is not a `Fold`: its members are not recomputable, its accumulators may be stateful and non-associative (rolling windows), and a context update must not invalidate what was accumulated.
It is three stages and two kinds of connector, which is also how it is structured today, with the context frontier derived from the partition rather than found by hand:

```python
per_chunk = Stage(pipeline, outputs=accumulator_keys, inputs=dynamic_keys)
context_frontier = Stage(pipeline, outputs=per_chunk.frontier, inputs=context_keys).dynamic_outputs
context_stage = Stage(pipeline, outputs=context_frontier, inputs=context_keys)
chunk_stage = Stage(pipeline, outputs=accumulator_keys, inputs=dynamic_keys + context_frontier)
finalize_stage = Stage(pipeline, outputs=target_keys, inputs=accumulator_keys + context_targets)
context = Forwarder(); accumulators = {key: EternalAccumulator() for key in accumulator_keys}
```

`stream_test.py` runs this shape, including a context update with the accumulator kept; the rewrite of the real class against its 35 tests is the next step, not done here.

`set_context` calls the context stage and pushes into the forwarder; `accumulate` calls the chunk stage with a chunk and the forwarder's value and pushes into the accumulators; `finalize` calls the third stage.
The accumulator classes, the key-set validation, `on_finalize`, `clear`, and `visualize` stay; `Accumulator.push` histograms by default through `maybe_hist`, which moves to the reducing subclasses so that a forwarder does not.
`allow_bypass` becomes "a dynamic key that is also an input of the finalize stage".
The module shrinks to the policy, which is what scipp/ess#732 said should stay: which values are transient and which are held.

### 6. What this means for essapps

- D8: the wrapper is `Stage(inputs=cheap_parameters)`.
  The cheap parameters stay a declaration, since they decide what a UI offers as a slider; the cached nodes are derived from it, not declared.
- D15: contribute, combine, finalize are `Fold`'s three entry points; the contribution is the dict at the cut keys.
  The declaration of which parameters finalize reads is derived from the graph and can be removed from D13/D15.
  A chained series is `fold.combine([previous, fold.contribute(row)])`; the in-memory fold is `compute` with the contributions held.
- Phase 3: the session model's warm workflow, the checkpoint model's in-application workflow, and the split model's two stages are the same `Stage` objects; the models differ only in where the objects live and when a cut value becomes a record, which is where a forwarder sits.
  The decision the stateless note defers is then about placement, not about a mechanism.
- Hierarchy inside one record (Bifrost angle groups, NMX chunks) is an inner fold whose finalize output is a member of the outer one, which is what D15 already says the callable does itself.

## Why this is not nested workflows

The nested-workflow idea that was rejected before sciline put a graph inside a node: sub-workflows with their own parameters, composed by an outer workflow.
Its problems were plumbing and opacity: parameters had to be passed through boundaries, and a boundary hid what was inside.

Here the author writes one flat graph, and stages are derived from it at the time they are used, from the keys the caller names.
Every parameter is set on the flat pipeline and reaches every stage; `fold[key] = value` is that, with the rebuild derived.
No provider or key is added to the author's graph.
Composition across stages is function composition in Python with connectors between, so the graph never contains a graph, and what is held is visible as objects.

An earlier draft kept a bridge, `as_pipeline()`, that synthesized providers for the cut keys so that the `with_*` helpers could keep returning a pipeline.
It was dropped: it was the only way two folds could compose, so it was not a bridge but the mechanism; it forced the member table to be fixed at construction; and it reintroduced the opacity, a provider that runs a loop with no per-member errors, progress, or visualization.
The trap to keep out of stays: a provider written by hand that runs a pipeline.

## Evidence

Prototype: `stage-prototype/stage.py`, tests in `stage-prototype/stage_test.py` and `stream_test.py`, run against sciline `main` (the pep695 branch needs an unreleased cyclebane for generics; on it all tests but the generics one pass as well).
It reads `TaskGraph._graph` for the concrete graph and hardcodes the naive scheduler inside stages; both go away inside v2.

Covered:

- static part computed once, dynamic part per call; an intermediate input cuts its ancestors; an input the outputs do not need is refused; an output that is an input is passed through; `warm` computes shared static work once;
- the warm workflow shape (cheap parameter after an expensive load);
- a fold with two cut keys and different combine functions equal to a manual loop; a two-column member table; a cut key independent of the members passed through; a group with no dependent cut key refused; contribute, chained combine, finalize as separate calls; groupby via pandas;
- adding a member costs one contribution; replacing one drops its contribution; a parameter after the cut keeps contributions; one before the cut drops them and, with `keep_members`, keeps the loaded members; a member key cannot be set as a parameter; a table with unknown columns is refused;
- banks over folded runs with the runs loaded once; a fold over a generic key with `TypeVar`-instantiated providers; two groups on one pipeline through `compute` and through the three entry points; the fold as a snapshot under later mutation of the caller's pipeline; per-member values of a key after the cut;
- the `StreamProcessor` shape: chunk stage, context in a forwarder, histogram in a reducer, context update without clearing, finalize.

LoKI validation (`stage-prototype/loki_validation.py`): the esssans multi-run test workflow, one mask file, two sample runs, two background runs.
Reference: `with_pixel_mask_filenames`, `with_sample_runs`, `with_background_runs` with `map`/`reduce`.
Prototype: one `Fold` with two groups on the flat pipeline, the masks as a list parameter.
`BackgroundSubtractedIofQ` and `BackgroundSubtractedIofQxy` are identical to the reference under `assert_identical`; per-member `NormalizedQ` equals both a single-run compute and `compute_mapped`; contribute, chained combine, finalize equal `compute`.
Provider call counts equal the reference's, including one read of the mask file, once `warm` computes the static parts of the two groups together; before that the file was read once per group.
Wall time is 6.5 s against 6.9 s under the same scheduler; under sciline's default dask scheduler the reference is about 1.7 s faster because the single graph runs the two sample runs in threads, and that parallelism is the driver's here, not in the prototype.
Adding the second sample run after computing with one costs one contribution: one more `apply_pixel_masks` call and no second mask read.

One finding on memory policy: with `keep_members=True`, changing `QBins` reruns everything from `apply_pixel_masks` on, and the wall time is the same as a cold compute.
The member frontier is derived, and in this graph it lies above the wavelength conversion, because that reads `WavelengthBins`, a parameter; what the graph computes from a member alone is small.
Holding more would need a second tier, parameters declared as rarely changing, which is `StreamProcessor`'s context and not in `Fold`.

Not shown by the prototype: the `StreamProcessor` rewrite against its real tests, and parallel members.

## Migration

1. sciline: `Pipeline` without map/reduce, `Stage`, `warm`, `provide`.
   Whether this is `sciline.v2` or the next major release is a real choice.
   The namespace lets esslivedata and external users pin through the change and keeps v1 importable next to v2; it also bundles two independent changes under one name, invites mixing v1 and v2 pipelines in one process with obscure failures, and guarantees a second rename.
   Recommendation: one major release, with the ESS monorepo migrated in one PR and esslivedata pinning the previous version until it follows; the last minor release before it deprecates `map`, `reduce`, and `constraints=`.
2. ess.reduce: `Fold`, `Forwarder`, and the accumulators in one module of stage-derived tools; `StreamProcessor` rewritten on `Stage` against its existing tests; `BatchProcessor` loses its mapped/unmapped fallback.
   `parameter_mappers`, which maps a list-valued parameter to a `with_*` helper returning a pipeline, is replaced by a registry from member key to cut, so that the UI builds a `Fold` over the pipeline and calls `set_members` for the list-valued parameters; `get_parameters` is unaffected because it runs on the flat pipeline.
3. esssans, essreflectometry, essspectroscopy, essnmx, essdiffraction: the `with_*` helpers that fold are replaced by a dedicated object per package, a `Fold` with the package's cuts, on which users set runs and parameters and compute; notebooks and tests change accordingly.
   `with_pixel_mask_filenames` in esssans and essdiffraction becomes a `PixelMaskFilenames` list parameter with two providers.
   `with_banks`, which maps without reducing, becomes a loop setting `NeXusDetectorName` on the fold or the pipeline, which is what its callers do with `compute_mapped` anyway.
4. esslivedata: the bifrost bank fold becomes a `Fold` computed before the `StreamProcessor` is built, its result set as a parameter of the processor's pipeline.
5. essapps: D8 and D15 wording as above; the D3/D6 spike's fake workflow with two accumulation points is a `Fold`.

## Open questions

- **Combine protocol.**
  `Fold` uses n-ary functions over held contributions, as `reduce(func=)` does today: `sc.concat` over a list is one pass where pairwise folding of binned events reallocates per step, and associativity is a property the workflow declares for D15, not a signature.
  `StreamProcessor` uses accumulator objects, which subsume the n-ary function and add non-associative policies.
  Both fit the connector slot; whether `Fold` should accept an accumulator per cut key instead, so that one protocol serves both drivers, is open.
- **Member labels.**
  The prototype takes a dict of columns, labeled by position, or a DataFrame, labeled by its index, and keeps held values by label when the row is unchanged.
  Recommendation: `Mapping[label, Mapping[Key, value]]` only, which `df.to_dict('index')` produces, so that pandas stays out of the library and the label is always the caller's.
- **What is held.**
  Contributions per member always, in the prototype; the member frontier as an option; nothing else.
  For event data both are large, and the LoKI finding above says the member frontier buys little without a second tier of rarely changing parameters.
  Recommendation: keep contributions, since they are what makes add and remove cheap and what D15 stores anyway; decide on the member frontier when a case shows it pays.
- **`contribute` and sibling groups.**
  `contribute(row)` warms only its own group's stages, so two groups contributed separately each read a shared static file.
  Warming all groups would load what another group's static part needs, wrong for a throwaway contribute process.
  Left as is; the driver that wants sharing calls `compute`.
- **Names.**
  `Stage`, `Fold`, `over`, `cut`, `contribute`/`combine`/`finalize`, `Forwarder` here; essapps says contribution, accumulation point, stage output.
  One vocabulary across sciline, ess.reduce, and essapps is worth settling before the code lands.
- **Parallel members.**
  An executor argument on `Fold`, or leave it to the caller with dask delayed over `contribute`.
