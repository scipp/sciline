# Map and reduce outside the graph

Proposal, with a prototype. Companion to the demand-driven generics design document on the `235-pep695-single-model-prototype` branch (scipp/sciline#237), which it supersedes in one respect: the open questions there that came from `map`/`reduce` disappear if `map`/`reduce` do.

## Summary

Sciline's `map`/`reduce` put a loop over members and a combine step inside the graph.
That has been tried twice (parameter tables in 23.08, cyclebane in 24.06), is the source of every breaking change and every remaining open question in the PEP 695 work, and still cannot express what `ess.reduce.streaming.StreamProcessor` and the essapps architecture need: run one part of a graph repeatedly with some values supplied per call, combine the results outside, run the rest once.

The proposal is to make that operation the primitive and build everything else from it:

- `sciline.v2.Pipeline`: the flat graph from type hints, with PEP 695 generics, without `map`, `reduce`, `groupby`, `constraints`, or cyclebane.
- `Stage`: the part of a pipeline from a set of input keys to a set of output keys, with everything that does not depend on the inputs computed once and the inputs supplied per call. This is what `StreamProcessor` builds by hand today, what scipp/sciline#241 asks for, and what the essapps warm workflow (D8) and split workflows (phase 3) are.
- `Fold`: a member table and a set of cut keys with a combine function per key; two stages of the same flat pipeline, `contribute` (member keys to cut keys) and `finalize` (cut keys to outputs). This replaces every real use of `map`/`reduce` in the ESS packages and is the essapps contribute/combine/finalize triple (D15).

A 330-line prototype on sciline `main` passes 17 tests covering the use-case shapes found in the ESS packages, reproduces the LoKI multi-run reduction identically against the existing `with_sample_runs` with the same provider call counts, and has been through one critical review whose findings are folded in below.

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
| Fold: map over a table, reduce at one or more keys, result feeds further providers | esssans runs (4 cut keys from one map, `merge_contributions`), essreflectometry runs (up to 7 cut keys, concat and `_any_value`), isissans zoom monitors (concat along a new dim, assert-unique position), bifrost triplets (two cut keys at different depths), masks in esssans and esspowder (dict union), NMX panels and MTZ files, DREAM detectors with a two-column table | events by concat, dense by sum, metadata by pick-one, dicts by union, `DataGroup` by key |
| Per-member results, no reduce | esssans `with_banks`, LoKI notebook, `BatchProcessor`, bifrost test | none; read with `compute_mapped` |
| Sequential composition on one pipeline | sample runs then background runs; banks over already-folded runs | as above |
| Fold in the static part of a `StreamProcessor` | esslivedata bifrost `EmptyDetector` over banks | `_combine_banks` |

Two things stand out.
Reduced nodes are almost always grafted back onto the same pipeline (`workflow[K] = workflow[K].map(df).reduce(func=f)`), so the caller sees an ordinary pipeline afterwards; `parameter_mappers` in `ess.reduce` depends on that.
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
```

Semantics:

- The graph is the ancestors of `outputs`.
  The dynamic part is `inputs` and their descendants; the static part is the rest.
  An input may be a parameter or an intermediate node; either way its own providers and ancestors are cut.
- The frontier, the static keys read by dynamic nodes plus static outputs, is computed on first use and held.
  Nothing else of the static part is kept.
- A call supplies exactly the inputs, computes only the dynamic part, and releases the supplied and computed values when it returns.
  This meets the three requirements of scipp/sciline#241 by construction: no rebuild, call-scoped lifetime, culling.
- A stage is a snapshot: the pipeline it was built from is not modified, and changing that pipeline afterwards does not affect the stage.
  To change a fixed parameter, build a new stage.
- `dynamic` and `dynamic_outputs` expose the partition: which nodes, and which outputs, depend on the inputs at all.
  Layered partitions, such as `StreamProcessor`'s static, context-dependent, and chunk-dependent sets, are built from these, see below.

What it replaces: `_build_streaming_workflow`, `_FedWorkflow`, `_find_descendants`/`_find_parents`, and the pruning-by-assignment hack in `StreamProcessor`; the sciline wrapper of D8, whose cache is the frontier of `Stage(inputs=cheap_parameters)`; and each half of a split workflow.

It belongs in sciline because it needs the concrete graph and `Provider` to do without private access, which is the argument scipp/sciline#241 makes.

### 3. `Fold`

```python
fold = Fold(
    pipeline,
    members=pd.DataFrame({Filename[SampleRun]: files}).rename_axis('run'),
    at={NormalizedQ[SampleRun, Numerator]: merge_contributions,
        NormalizedQ[SampleRun, Denominator]: merge_contributions},
    outputs=(BackgroundSubtractedIofQ,),
)
fold.compute(BackgroundSubtractedIofQ)      # contribute per member, combine, finalize
fold.compute_members(NormalizedQ[SampleRun, Numerator])   # per-member values
fold.contribute(run); fold.combine([...]); fold.finalize(contribution)   # the three stages
fold.as_pipeline()      # a flat pipeline whose cut keys are provided by the fold
```

Semantics:

- `members` is a table: one row per member, one column per key; a dict of columns or a DataFrame.
  The row index is the member key.
- `at` maps cut keys to n-ary combine functions, today's `reduce(func=)` signature, which every combine in the ESS packages already has; `outputs` defaults to the pipeline's output keys, since the `with_*` helpers reduce their keys without naming an output.
  A cut key that does not depend on the member keys is not folded; `finalize` computes it from the fixed part of the graph.
  That removes essreflectometry's `try/except`; it also means a misplaced cut key is silently static, which `cut` reports and a test should check.
- The fold takes a copy of the pipeline when it is built, so that every stage, `compute_members`, and `as_pipeline()` see one snapshot however the caller's pipeline is changed afterwards.
- `contribute` is `Stage(outputs=cut, inputs=member_keys)`, `finalize` is `Stage(outputs=outputs, inputs=cut)`, both from the same flat pipeline, so a parameter is set once and reaches both.
  Which parameters `finalize` reads is derived: the ones that are not ancestors of the cut.
- A contribution is a dict at the cut keys.
  `combine` applies the per-key function across contributions; `finalize` takes one contribution.
  The three can run in three processes with the contribution serialized between them.
- `as_pipeline()` returns the flat pipeline with the cut keys provided by synthesized providers whose inputs are the contribute stage's frontier.
  A change to any parameter upstream of the cut reruns the fold; the member keys and what only they reach are pruned from the graph, so a later assignment to a pruned key is a no-op, as it is for a mapped key today.
  This is the drop-in form for `with_sample_runs` and the other `with_*` helpers, and what `parameter_mappers` needs; `get_parameters` is unaffected because it runs on the unfolded pipeline and `assign_parameter_values` folds afterwards.
  The synthesized node needs a key that is unique per cut, stable across processes, and picklable; the prototype uses a type named after the cut keys, which is enough for sibling folds under the dask scheduler but not for `serialize`, and v2's `provide(key, callable)` should take a plain hashable key instead.

What is not in `Fold`:

- `groupby`: a pandas `groupby` over the member table and a fold per group.
- Hierarchy: a fold over a folded pipeline (banks over runs), or a fold whose members are themselves folded (angle groups inside a run, runs across records).
  Both are compositions of plain objects; nothing nests inside a graph.
- Parallelism over members: a driver concern, threads or dask over `contribute`, not a property of the graph.
  The single-scheduler run over all members that in-graph mapping gave is what this loses; it is also what made everything else hard.
  Sharing of member-only work across composed folds is kept by the member frontier (see the LoKI validation).

### 4. `StreamProcessor` on `Stage`

`StreamProcessor` is not a `Fold`: its members are not recomputable, its accumulators may be stateful and non-associative (rolling windows), and a context update must not invalidate what was accumulated.
It is three stages, which is also how it is structured today, with the context frontier derived from the partition rather than found by hand:

```python
per_chunk = Stage(pipeline, outputs=accumulator_keys, inputs=dynamic_keys)
context_frontier = Stage(pipeline, outputs=per_chunk.frontier, inputs=context_keys).dynamic_outputs
context_stage = Stage(pipeline, outputs=context_frontier, inputs=context_keys)
chunk_stage = Stage(pipeline, outputs=accumulator_keys, inputs=dynamic_keys + context_frontier)
finalize_stage = Stage(pipeline, outputs=target_keys, inputs=accumulator_keys + context_targets)
```

`stream_test.py` runs this shape, including a context update with the accumulator kept; the rewrite of the real class against its 35 tests is the next step, not done here.

`set_context` calls the first and holds the result; `accumulate` calls the second with a chunk and the held context and pushes into accumulators; `finalize` calls the third.
The accumulator classes, the key-set validation, `on_finalize`, `clear`, and `visualize` stay; the node classification behind `visualize` is read off the stages' frontier and dynamic sets.
`allow_bypass` becomes "a dynamic key that is also an input of the finalize stage".
The module shrinks to the policy, which is what scipp/ess#732 said should stay: which values are transient and which are held.

### 5. What this means for essapps

- D8: the wrapper is `Stage(inputs=cheap_parameters)`.
  The cheap parameters stay a declaration, since they decide what a UI offers as a slider; the cached nodes are derived from it, not declared.
- D15: contribute, combine, finalize are `Fold`'s three methods; the contribution is the dict at the cut keys.
  The declaration of which parameters finalize reads is derived from the graph and can be removed from D13/D15.
  A chained series is `fold.combine([previous, fold.contribute(new)])`; the in-memory fold is the same call with the partial held.
- Phase 3: the session model's warm workflow, the checkpoint model's in-application workflow, and the split model's two stages are the same `Stage` objects; the models differ only in where the objects live and when a cut value becomes a record.
  The decision the stateless note defers is then about placement, not about a mechanism.
- Hierarchy inside one record (Bifrost angle groups, NMX chunks) is an inner fold whose finalize output is a member of the outer one, which is what D15 already says the callable does itself.

## Why this is not nested workflows

The nested-workflow idea that was rejected before sciline put a graph inside a node: sub-workflows with their own parameters, composed by an outer workflow.
Its problems were plumbing and opacity: parameters had to be passed through boundaries, and a boundary hid what was inside.

Here the author writes one flat graph, and `Stage` and `Fold` are derived from it at the time they are used, from the keys the caller names.
Every parameter is set on the flat pipeline and reaches every stage.
No provider or key is added to the author's graph, with one exception: `as_pipeline()` synthesizes providers for the cut keys, whose inputs are derived from the graph and are real type hints.
Composition across stages is function composition in Python, so the graph never contains a graph.

The strongest argument is the one the plumbing objection was about: every static key the dynamic part reads becomes a real, typed argument of the synthesized provider, so every settable parameter survives the fold and is set in one place.

What remains of the objection is opacity: inside `as_pipeline()` there are no per-member errors, no progress, and nothing to visualize.
So `as_pipeline()` is a bridge for the `with_*` helpers and the UI's `parameter_mappers`, not the way to use folds; new code, essapps included, uses `Fold` directly.
The trap to keep out of: a provider written by hand that runs a pipeline; `as_pipeline()` should stay the only one, and its callers should migrate.

## Evidence

Prototype: `stage-prototype/stage.py`, tests in `stage-prototype/stage_test.py` and `stream_test.py`, run against sciline `main` (the pep695 branch needs an unreleased cyclebane for generics; on it all tests but the generics one pass as well).
It reads `TaskGraph._graph` for the concrete graph, hardcodes the naive scheduler inside stages, and synthesizes providers with `exec`; all three go away inside v2.

Covered:

- static part computed once, dynamic part per call; an intermediate input cuts its ancestors; an input the outputs do not need is refused;
- the warm workflow shape (cheap parameter after an expensive load);
- a fold with two cut keys and different combine functions equal to a manual loop; a two-column member table; a cut key independent of the members passed through; contribute, chained combine, finalize as separate calls; groupby via pandas;
- `as_pipeline()`: a flat pipeline, the member key pruned from the graph, rerun on a change upstream of the cut without reloading the members; sibling folds on one pipeline under the dask scheduler; the fold as a snapshot under later mutation of the caller's pipeline;
- banks over folded runs; a fold over a generic key with `TypeVar`-instantiated providers; an intermediate input with a shared ancestor, where the rest is frozen;
- the `StreamProcessor` shape: chunk stage, held context, context update without clearing, finalize.

LoKI validation (`stage-prototype/loki_validation.py`): the esssans multi-run test workflow, one mask file, two sample runs, two background runs, three folds composed on one pipeline (masks, sample runs, background runs) as `with_pixel_mask_filenames`, `with_sample_runs`, and `with_background_runs` do it with `map`/`reduce`.
`BackgroundSubtractedIofQ` and `BackgroundSubtractedIofQxy` are identical to the reference under `assert_identical`, per-member `NormalizedQ` equals both a single-run compute and `compute_mapped`, and contribute, chained combine, finalize equal the one-shot result.
Provider call counts equal the reference's, including one read of the mask file, and wall time is 7.2 s against 7.4 s for the reference under the same scheduler.
Under sciline's default dask scheduler the reference takes 5.7 s, because the single graph runs the two sample runs in threads; in this design that parallelism is the driver's, a thread pool over `contribute`, and is not in the prototype.

The call counts needed one addition, and it is the same trick `StreamProcessor` uses for its static inputs, applied per member.
A contribute stage has two kinds of inputs: the member keys and the outer frontier.
Nodes that depend on the member keys alone (reading a mask file, loading a run) are propagated once per member up to the first node that also reads a frontier value, the member frontier, and held there; a call with new frontier values computes only from the member frontier down.
Without this, a fold composed under another fold reran its whole contribute per outer member, and the mask file was read once per sample run.
Holding the member frontier is a memory policy: for a fold over runs it is the loaded run, which a graph also holds for the duration of one compute but not across computes.
The prototype holds it for the fold's lifetime; the eventual `Fold` needs this as an option, with releasing after each combine as the default for large members.

Two more things the validation changed in the prototype: a stage computes its static values on first use rather than at construction, since a fold used only through `as_pipeline()` never needs its finalize stage, and `outputs` defaults to the pipeline's sinks, since the `with_*` helpers reduce their keys without naming an output.

Not shown by the prototype: the `StreamProcessor` rewrite against its real tests, and parallel members.

## Migration

1. sciline: `Pipeline` without map/reduce, `Stage`, `provide`.
   Whether this is `sciline.v2` or the next major release is a real choice.
   The namespace lets esslivedata and external users pin through the change and keeps v1 importable next to v2; it also bundles two independent changes under one name, invites mixing v1 and v2 pipelines in one process with obscure failures, and guarantees a second rename.
   Recommendation: one major release, with the ESS monorepo migrated in one PR and esslivedata pinning the previous version until it follows; the last minor release before it deprecates `map`, `reduce`, and `constraints=`.
   `Stage` belongs in sciline whichever way this goes, since it needs the graph and `Provider`.
2. ess.reduce: `Fold` next to `StreamProcessor` in one module of stage-derived tools, since it is where the combine functions and the essapps wrapper live and where it can change without a sciline release; `StreamProcessor` rewritten on `Stage` against its existing tests; `with_*` helpers in esssans, esspowder, essreflectometry, essspectroscopy return `fold.as_pipeline()` so their callers do not change; `BatchProcessor` loses its mapped/unmapped fallback; `parameter_mappers` unchanged.
   `with_banks`, which maps without reducing, becomes a helper returning one pipeline per bank, which is what its callers do with `compute_mapped` anyway.
   NMX and DREAM notebooks use `Fold` directly.
3. esslivedata: the bifrost bank fold becomes `Fold(...).as_pipeline()` before the `StreamProcessor` is built, as now.
4. essapps: D8 and D15 wording as above; the D3/D6 spike's fake workflow with two accumulation points is a `Fold`.

## Open questions

- **Combine protocol.**
  N-ary, as in the prototype and in `reduce(func=)` today: a pairwise call is the n-ary one with two arguments, `sc.concat` over a list is one pass where pairwise folding of binned events reallocates per step, and associativity is a property the workflow declares for D15, not a signature.
  Accumulators stay in `StreamProcessor`.
- **Member table type.**
  The prototype accepts a dict of columns or a DataFrame.
  Recommendation: `Mapping[member, Mapping[Key, value]]` only, which `df.to_dict('index')` produces, so that pandas stays out of the library and there is one convention for member labels.
- **What is held per member.**
  Two candidates: the member frontier (so that a rerun with changed upstream parameters does not reload), and the contribution (so that D15's removal of a member is a combine over the rest).
  Both are memory policies, and for event data both are large; recommendation: options on `Fold`, off by default, on in a warm session.
- **`as_pipeline()`.**
  Keep it for the `with_*` helpers, or migrate their callers to `Fold` and drop it.
  Recommendation: keep it; it is forty lines and it is what keeps the change non-breaking for `parameter_mappers` and the UI.
- **Names.**
  `Stage`, `Fold`, `cut`, `members`, `contribute`/`combine`/`finalize` here; essapps says contribution, accumulation point, stage output.
  One vocabulary across sciline, ess.reduce, and essapps is worth settling before the code lands.
- **Parallel members.**
  An executor argument on `Fold`, or leave it to the caller with dask delayed over `contribute`.
