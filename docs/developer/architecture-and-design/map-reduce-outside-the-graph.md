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
The code that needs repetition, the *driver*, cuts the pipeline into *stages*, calls them from an ordinary Python loop, and places objects between the stages that hold state:

```text
            ┌──────────────┐        ┌─────────────┐        ┌──────────────┐
 inputs ───▶│   stage A    │──push─▶│  connector  │─value─▶│   stage B    │───▶ outputs
 per call   └──────────────┘        └─────────────┘        └──────────────┘
                                 (accumulator or forwarder)
```

Sciline provides the stages and the `Accumulator` protocol with two accumulator factories, `Buffered` and `Reduced`.
Everything else, in particular what is kept between calls and when it is discarded, is decided by the code that runs the loop.

Terms used in this document:

| Term | Meaning |
|---|---|
| stage | The part of a pipeline from a set of input keys to a set of output keys (`sciline.Stage`). |
| static part, dynamic part | The part of a stage's graph that does not depend on the inputs, and the part that does. The docstrings and the user guide call them the held part and the per-call part. |
| frontier | The keys of the static part that the dynamic part reads, plus outputs that do not depend on the inputs. A stage holds the values at these keys. |
| connector | An object between stages with `push(value)` and `value`. |
| accumulator | A connector that combines all pushed values (`sciline.Accumulator`). |
| forwarder | A connector that holds the last pushed value until the next push. Lives in ess.reduce. |
| member | One repetition, for example one run, given by the values of the inputs of a stage. |
| accumulation key | A key at which the values of the members are combined. |
| contribution | The values at the accumulation keys for one member, as a dict. |
| contribute stage | A stage from the inputs that give a member to the accumulation keys. |
| finalize stage | A stage from the accumulation keys to the outputs. |
| loop | One level of the driver's nested loops, given by the keys whose values the driver supplies on each iteration. |
| outer stage | The stage of a loop that computes, once per iteration, the values that the stages inside the loop read and that depend on the inputs of the loop. |
| forwarded value | A value that an outer stage returns and the stages inside its loop take as input. |
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
  A call supplies a value for each input and for nothing else.
  It computes only the dynamic part, reading the held frontier values, and returns the outputs.
  The supplied and intermediate values are released when the call returns.
- **Pass-through.**
  An output that is also an input is returned as supplied.
  This makes a stage from a key to itself the identity, which a driver needs when its members are already values at its accumulation keys, such as results of other loops.
- **Snapshot.**
  The stage is built from the task graph of the pipeline at construction time.
  Later changes to the pipeline do not affect it; to change a parameter, build a new stage.
  Parameter values are held by reference, not copied, so a value modified in place, rather than set anew, can change what the stage computes.
- **Introspection.**
  `dynamic` lists the keys that depend on the inputs, and `dynamic_outputs` the outputs among them.
  `keys` lists every key the stage uses.
  A parameter that is not an input is in `keys` exactly when changing it on the pipeline would change the stage's results.
  Callers use this to decide what to rebuild (section 6.7).
- **`warm(*stages)`.**
  Stages built from one pipeline often share static work, for example a file that each of them reads.
  `warm` computes the static parts of several stages in one scheduler run, so shared intermediate results are computed once and then released.
  Each stage keeps only its own frontier values, as if it had been warmed alone.
- **Threads.**
  A stage can be called from several threads at once; the static part is still computed only once.

### Implementation notes

The stage builds its task graph with `to_task_graph` and `HandleAsComputeTimeException`.
Keys without a value, such as the inputs, thus become nodes that fail only if they are computed, instead of failing when the stage is built.
The scheduler default is the same as for `Pipeline` (section 8.7).

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
  `func` must be associative and must not modify its arguments, because the pushed values belong to the driver.
  Since the update is not in place, a sum briefly holds the old result, the new result, and the pushed value.
  An accumulator that updates in place has to be written by the workflow author, who then owns the copy of the first value; the accumulators in ess.reduce are such objects.

For concatenation, `Buffered` and `Reduced` cost about the same memory; for a sum of large dense arrays, a running result holds one array instead of one per member.
Both are factories: each call returns a new, empty accumulator, and a driver makes new accumulators for each computation.
An accumulator kept from an earlier computation adds the new members to the old result, without an error.

### Combining combined values

A driver may push combined values into a new accumulator, to combine in groups or as a chain, for example one new member at a time onto a previous result, or values combined in separate processes.
This requires two things of an accumulator:

1. What `value` returns can be pushed again.
2. The result does not depend on how the pushes were grouped.

`Buffered` satisfies both if its function is associative, that is `func(func(a, b), c) == func(a, b, c)`.
`Reduced` requires associativity anyway.
An ess.reduce accumulator satisfies both if its `push` accepts what its `value` returns.
The histogramming accumulators must keep this property once `maybe_hist` moves out of their base class (see below).

### Relation to the ess.reduce accumulators

An accumulator over runs and an accumulator in a stream are the same kind of object; whether the values come from many members or from one input over time only matters to the driver.
The driver also owns their lifetime (`clear`), their identity (contributions by label), and whether the result may depend on the order of pushes.

The accumulators of `ess.reduce.streaming` satisfy the protocol once `maybe_hist`, which histograms in the base class `push`, moves to the subclasses that need it.
The base class then adds nothing to the protocol, and a forwarder built on it does not histogram.
Their `clear` method stays an ess.reduce convention; no base class beyond the protocol is shared.
`Accumulator` is in sciline because `Buffered` and `Reduced` return it and drivers use it to type their accumulators; `Forwarder` stays in ess.reduce because nothing in sciline returns or consumes one.

## 5. Nested loops

Banks within runs, or chunks of a stream within a context, are loops in loops.
The stage of the inner loop reads values that depend on the outer loop only, such as the content of the file of a run.
These should be computed once per iteration of the outer loop, not once per iteration of the inner loop.
A stage built for the inner loop alone holds them at its frontier, for the one run set on the pipeline.
The driver therefore builds one stage per loop, and the outer stage computes the *forwarded values* that the inner stage takes as inputs.
They are the values at the frontier of the inner stage that depend on the inputs of the outer loop:

```python
bank_stage = Stage(pipeline, inputs=(Bank,), outputs=(Numerator, Denominator))
# What bank_stage holds that depends on the run:
forwarded = Stage(pipeline, inputs=(Filename,), outputs=bank_stage.frontier).dynamic_outputs
run_stage = Stage(pipeline, inputs=(Filename,), outputs=forwarded)
bank_stage = Stage(pipeline, inputs=(*forwarded, Bank), outputs=(Numerator, Denominator))
final_stage = Stage(pipeline, inputs=(Numerator, Denominator), outputs=(IofQ,))
```

Values at the frontier that do not depend on the run, such as a calibration, stay held by the bank stage.
A driver for runs times banks:

```python
warm(run_stage, bank_stage, final_stage)
acc = {Numerator: Buffered(concat)(), Denominator: Reduced(add)()}
for filename in filenames:
    held = run_stage.compute({Filename: filename})
    for name in bank_names:
        out = bank_stage.compute({**held, Bank: name})
        for key in bank_stage.dynamic_outputs:
            acc[key].push(out[key])
result = final_stage.compute({k: a.value for k, a in acc.items()})
```

With several stages inside a loop, such as the triplet stage and the per-run step of Bifrost (section 6.3), the run stage outputs the forwarded values of all of them, and each stage takes those at its own frontier.
`Stage.compute` accepts exactly the inputs of the stage, so the driver selects the values for each stage, `{k: v for k, v in held.items() if k in stage.inputs}`.

Building the stages of nested loops this way has two pitfalls:

- **An output of the inner stage that depends on the outer loop only.**
  A key that depends on the run but not on the bank, declared as an output of the bank stage, is held, so it is at the frontier and forwarded.
  The rebuilt bank stage takes it as input and passes it through, so it is in `dynamic_outputs`, and the driver pushes it once per bank.
  The combined value is wrong without any error.
  Run-level accumulation keys are outputs of the run stage instead.
- **A stage that is not rebuilt.**
  A stage inside the run loop that still holds a value depending on `Filename` uses the one run set on the pipeline, or fails if none is set.
  `warm`, given all stages of a driver, rejects a stage that holds a value depending on a parameter that another stage takes as input.
  This relies on the driver warming its stages together, which it does anyway so that shared work is done once.

A function that derives these stages, such as `enclose`, is a likely later addition.

### What is deliberately not in sciline

- **Loops, held values, and accumulators.**
  The driver decides what is held, for how long, and when accumulators are made and discarded, so memory is bounded by what it holds, for example one run.
- **Parallelism over members.**
  A stage call is a plain function call, so the driver can map it over members with threads, processes, or dask.
  This is the one thing that computing everything in one graph provided for free.
- **Parameter changes.**
  Parameters are set before the stages are built, and the stages are snapshots; a changed parameter means new stages.

## 6. Composition

Every use from section 1 is a combination of stages, connectors, and a driver.
This section shows each.

### 6.1 A package object (esssans)

esssans users today set parameters, call `with_sample_runs` and `with_background_runs`, and compute.
To keep this experience, esssans returns its own object instead of a pipeline.
The validation script contains such an object, `SansReduction`, in about 50 lines.
It is built from a pipeline with all parameters set, so setting the runs is the last step of the setup.
To change a parameter, the user builds a new object; nearly every parameter is read per run, so this costs no more than rebuilding only the affected stages would.
It holds one contribute stage per run type (sample and background), one finalize stage from the accumulation keys of both to the outputs, and the contributions of each run type by filename.
`set_runs(run_type, runs)` records the runs and drops the contributions of runs no longer listed.
`compute()` warms all stages together, contributes the runs that have no contribution yet, pushes the held contributions into new accumulators, and finalizes.

The background stage reads `DetectorMasks`, which in esssans depends on the detector IDs of the sample run set on the pipeline.
With several sample runs, this run is not defined.
`warm` over both stages rejects it: the background stage holds a value that depends on `Filename[SampleRun]`, which the sample stage takes as input.
Forwarding the masks from the sample loop is not the fix, because the background loop is not inside the sample loop; nested in it, each background run would be computed once per sample run.
The validation script takes the detector IDs from the empty-beam run instead, so the masks depend on no run and are computed once (section 11).

### 6.2 Several loops, one final stage

Sample runs and background runs are two loops on the same pipeline, and a final stage outside both loops takes the accumulation keys of both as inputs.
All stages are warmed together, so work they share, such as reading a mask file, is done once.

### 6.3 Runs times banks

Mapping over detector banks and runs at once needs two levels.
esssans tests this shape with `with_banks` over `with_sample_runs`; LoKI and Bifrost with several runs need it too.
A flat loop over (run, bank) pairs computes run-level work, such as loading monitors, once per bank, and pushes a run-level accumulation key once per bank; a loop per bank over the runs loads every run once per bank.
With one stage per loop, the run stage computes the per-run values once per iteration of the run loop, and the bank stage takes them as inputs (section 5).

Bifrost merges the triplets of one run into one detector and concatenates the runs.
The per-run step after merging is a second stage inside the run loop, from the accumulation keys of the triplet loop:

```python
triplet = Stage(pipeline, inputs=(NeXusDetectorName,), outputs=(EmptyDetector, NeXusData))
run_final = Stage(pipeline, inputs=(EmptyDetector, NeXusData), outputs=(NormalizedDetector,))
final = Stage(pipeline, inputs=(NormalizedDetector,), outputs=(EnergyQDetector,))
```

The run stage computes what either stage reads from the run: `NeXusFile` and `PrimaryGraph` for the triplets, and the angles, monitor, and proton charge for `run_final`.
Each file is therefore opened once per run.
Work that depends on the bank alone, such as LoKI's per-bank lookup table, is computed once per run and bank, unless the driver holds it across runs.
Groups within a run, such as angle groups in Bifrost, are a further level, and nothing is nested inside a graph.

### 6.4 A reduce inside per-member work (esssans pixel masks)

In esssans, `DetectorMasks` reads the detector IDs of the sample run, so with map/reduce the graph computes the mask fold once per sample run.
This is not a loop with an accumulator, because the combined value is needed inside the work for each member.
Outside the graph it becomes a list parameter, `PixelMaskFilenames`, and two providers: one reads all mask files (static and shared by all runs), and one builds the masks from the detector IDs.
With detector IDs that do not depend on the run (section 6.1), the masks are static as well.
Loops over members are meant for members that are expensive to compute or whose individual results users want to see; a handful of small files combined by union is neither.

### 6.5 Passing a result from one driver to another

A result enters another driver as an ordinary parameter of the flat pipeline; the bifrost bank fold in esslivedata becomes:

```python
banks = Stage(pipeline, outputs=(EmptyDetector,), inputs=(NeXusDetectorName,))
acc = Buffered(combine_banks)()
for name in bank_names:
    acc.push(banks.compute({NeXusDetectorName: name})[EmptyDetector])
pipeline[EmptyDetector] = acc.value
processor = StreamProcessor(pipeline, ...)
```

### 6.6 `StreamProcessor`

A stream differs from a loop over runs in three ways.
Chunks of a stream cannot be recomputed.
The accumulators are reused between finalizations and need not be associative (rolling windows).
A change of context, such as a new detector position, must not discard what was accumulated.
`StreamProcessor` is a loop over contexts, with one chunk stage per group of dynamic keys and one finalize stage inside it:

```python
chunk_stages = [Stage(pipeline, inputs=keys, outputs=acc_keys)
                for keys, acc_keys in groups]    # accumulators by the dynamic keys they read
finalize_stage = Stage(pipeline, inputs=(*all_acc_keys, *bypass_keys),
                       outputs=accumulated_targets)
# Rebuilt as in section 5: the context stage computes, from context_keys,
# context_only_targets and what these stages hold that depends on the context.
```

`set_context` calls the context stage and holds its result in a forwarder.
`accumulate` calls each chunk stage whose inputs the chunk supplies and pushes into the accumulators.
`finalize` calls the finalize stage.
What the chunk and finalize stages read from the context, today found by `_find_descendants` and `_find_parents`, is derived from their frontiers.
A target that depends on the context alone is an output of the context stage, and `allow_bypass` becomes an explicit input of the finalize stage.
A target that depends on no input belongs to no loop; a plain `Stage` without inputs computes it.
The accumulator classes, the grouping of accumulators by dynamic keys, the validation of key sets, `on_finalize`, `clear`, and `visualize` stay.
The module shrinks to its policy, which values are transient and which are held, as scipp/ess#732 proposed.

With one context stage, an update of any context key recomputes all context-derived values, where `StreamProcessor` today recomputes only what depends on the updated keys.
In esslivedata this costs compute and changes no results.
The only reset tied to the context, the reset-on-move of `NoCopyAccumulator`, compares coordinate values on each push, and values recomputed from unchanged inputs are identical; nothing inspects which context nodes were recomputed.
The extra work is at most one rebuild of the detector projection when a rare key (an ROI edit, or a wavelength lookup table after a chopper change) is updated in a job with moving detector geometry.
The calls inside `StreamProcessor` must also honour esslivedata's choice of scheduler (section 8.7).

`test_stream_of_chunks_with_context_held_between_changes` in `tests/stage_test.py` runs this shape with stages built by hand, including a context update that keeps the accumulator.

### 6.7 essapps

essapps is the planned service for automatic, batch, and interactive data reduction.
Its design is on the `architecture-sketch` branch (`docs/developer/architecture.md`), and `docs/developer/stages.md` and `docs/developer/aggregation.md` there relate it to this proposal.
In that design a workflow is described by a *spec* that declares its parameters and outputs, every run produces a stored *record*, and a *binding* connects a spec to the sciline workflow.
A record holds only what was computed: the spec, every parameter value, and the outputs.
Stages and accumulators are how it is computed; a session holds them as caches, and they never appear in a record.
Only the binding knows the graph, so a spec does not say which parameters each part of the workflow reads.
The essapps interface follows what sciline and the package objects provide, so that no problem is solved twice.
It uses them in four places:

- **Tuning.**
  The caller names the stage as a *template*, whose blanks are the parameters that will move, and the session holds `Stage(pipeline, inputs=blanks, outputs=targets)` for it.
  A rerun after changing a blank reuses everything upstream; a blank that the targets do not need is dropped by the binding, because `Stage` refuses it.
  A change to any other parameter needs a new stage only if the parameter is in `stage.keys`.
- **A sum over runs.**
  A sum is one request whose run parameter is a list.
  The binding wraps the package object (section 6.1) and computes the sum in one process.
  Values that differ per run are further inputs of the contribute stage; nested levels stay in the package's driver (section 6.3).
  In a session, the contribute stage and the contributions by run are kept, so a call whose list extends the previous one contributes only the new runs.
  A long-lived runner that holds this state for one series under a rule is called a *fold* in essapps.
- **A sum spread over nodes.**
  The workflow author splits the sum into two specs, and the binding builds both from two stages of one pipeline:

  ```python
  contribute_stage = Stage(pipeline, inputs=(Filename[SampleRun],), outputs=ACC_KEYS)
  final_stage = Stage(pipeline, inputs=ACC_KEYS, outputs=(IofQ,))
  assert set(contribute_stage.dynamic_outputs) == set(ACC_KEYS)  # each varies per run

  def contribute(filename):   # CONTRIBUTE: one run -> accumulation keys, exposed as outputs
      return contribute_stage.compute({Filename[SampleRun]: filename})

  def combine(contributions): # COMBINE: references to the outputs of CONTRIBUTE -> result
      acc = {k: make() for k, make in ACCUMULATORS.items()}
      for contribution in contributions:
          for k, a in acc.items():
              a.push(contribution[k])
      return final_stage.compute({k: a.value for k, a in acc.items()})
  ```

  Each record describes only its own computation, so no check that the contributions agree is needed.
- **Interactive applications (phase 3).**
  essapps compares three models: a session that holds state, an application that holds state itself, and a stateless service that splits a workflow into two specs with the intermediate value stored as a record.
  In all three the objects are the same, a stage and a connector after it, so choosing a model is a question of placement, not of a new mechanism.

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
The parameter-tables guide is replaced by the guide on stages, and the generic-providers guide loses `constraints=`.

## 8. Design choices

### 8.1 Connectors as separate objects, not tiers inside `Stage`

The alternative was a `Stage` with several tiers of inputs and a policy per tier for what to hold, which hides the held values inside one class and puts policy into sciline.
One plain stage per loop gives the same boundaries, and every held value is an object the driver can inspect, clear, serialize, or send to another process: the explicit lifetime that scipp/sciline#241 asks for.
`Stage` holds the values at its frontier itself.
A stage without inputs followed by a forwarder would do the same.
It is built in because every stage needs it.

### 8.2 No generic network object (deferred)

A general object could hold stages and connectors and route pushes through them.
The real uses disagree on exactly this routing: a chunk runs immediately because it cannot be kept, a context update runs immediately but must not touch the accumulators, a loop over runs can run lazily, `clear` discards accumulators but keeps the context, and rolling windows and `on_finalize` add their own rules.
A generic object would either expose parameters for all of this or hide one choice, and it would be a graph of stages with its own scheduler, which is nested workflows one level up (section 8.8).
Each real use is a loop of under twenty lines over plain objects, and that loop is where the policy belongs.
If package objects turn out to repeat the same code, that repetition is the basis for a generalization of the package objects; if essapps phase 3 needs connectors on process boundaries, that is a placement layer over the same stages and connectors.

### 8.3 Boundaries between levels are derived

Three callers need the same boundary: the context frontier of `StreamProcessor`, runs times banks in esssans, and runs times triplets in Bifrost.
Chosen by hand, it fails silently: a run-level key left out is recomputed per bank, which costs time but gives the right result.
It is derived from the frontier of the stages inside the loop instead (section 5).

Stages are connected in two ways:

| Connection | From | To | Given by |
|---|---|---|---|
| forwarded value | outer stage | stages inside its loop | derived from the frontier |
| accumulation key | stages inside a loop | a stage after the loop | declared as outputs of one stage and inputs of the next |

The accumulation keys are declared, not derived: the combine happens outside the graph, so the graph cannot tell that a key such as `NormalizedDetector` comes after combining the triplets.

### 8.4 No state in the objects sciline provides

An earlier draft put member tables with groups, the held contributions, a held per-member frontier, and invalidation rules in `__setitem__` into an aggregation object (section 8.5).
It rebuilt the map/reduced pipeline in another form: to predict the cost of a parameter change, one had to know which of five held things read the key.

The per-member frontier was meant to avoid reloading runs when a parameter changes, but on LoKI it gave no measurable benefit: the wavelength conversion reads the parameter `WavelengthBins`, so the frontier lies upstream of the conversion.
The package object (section 6.1) therefore does not take parameter changes: a new parameter value means a new object.

A later variant held, within one computation over a flat runs-times-banks table, the work that depends on a single member key, once per distinct value.
What it held was decided by the shape of the graph, not by the author: if the per-bank work reads a whole loaded run, every run stays in memory until the table is done.
The driver in section 5 holds the same values for one run at a time, visibly.

### 8.5 No `Aggregation`

`Aggregation` packaged a contribute stage, accumulator factories, and a finalize stage, with `contribute`, `combine`, `finalize`, and `compute(table)`.
It was prototyped and dropped:

- No real driver used its finalize stage or `compute`: the SANS finalize reads the accumulation keys of two run types, the per-run finalize of Bifrost also reads run-level values, and nested drivers push inside their own loops instead of calling `combine`.
- `compute(table)` invited the flat runs-times-banks table, which counts a run-level accumulation key once per bank without an error.
- Its one guarantee, that a key not depending on the members is not accumulated, is `Stage.dynamic_outputs`.

Its accumulator factories remain as `Buffered` and `Reduced`, and their reason remains: a driver makes new accumulators per computation (section 4).

### 8.6 No bridge back into a pipeline

Another draft had `as_pipeline()`, which added providers for the accumulation keys so that the `with_*` helpers could keep returning a pipeline.
It was the only way to compose two aggregations, so it was the actual mechanism, not a convenience.
It fixed the members when the pipeline was built.
It hid a loop inside a provider, with no per-member errors, progress, or visualization.
The trap to avoid remains: a hand-written provider that runs a pipeline.

### 8.7 Default scheduler

`scheduler_or_default` lives in `sciline.task_graph` and looks up `DaskScheduler` in that module on each call, for pipelines and stages alike.
esslivedata replaces `sciline.task_graph.DaskScheduler` to select its scheduler, including for the calls inside `StreamProcessor` that it cannot pass a scheduler to; a sciline test checks that this changes the default of both.
A public way to set the default scheduler would be cleaner.
Until it exists, sciline keeps the replacement working for stages (section 10).

### 8.8 Not nested workflows

Nested workflows, sub-workflows with their own parameters inside a node of an outer workflow, were rejected earlier for plumbing (parameters passed across boundaries) and opacity (a boundary hid what was inside).
Here the author writes one flat graph, and stages are cut from it by the driver, from the keys the driver names.
Every parameter is set on the flat pipeline and reaches every stage, and composition across stages is ordinary Python, so no graph contains another graph.

### 8.9 Names

- **Accumulator**, **contribution**, and *contribute*/*combine*, as in Beam, Flink, and Spark, and in essapps.
  `fold` was avoided because it means reshaping in scipp (essapps uses *fold* for a long-lived runner, not for the operation).
- **Accumulation key**, matching the essapps term.
- `Forwarder` is the working name in ess.reduce; the final name is ess.reduce's decision.

## 9. Validation

### Unit tests

The implementation is in `src/sciline/stage.py` and `src/sciline/accumulators.py`, tested in `tests/stage_test.py` and `tests/accumulators_test.py`.
It runs on sciline `main`.
An earlier version of the stage tests was run on the generics branch and passed.

The tests cover:

- **Stage:** static part computed once and dynamic part per call; an intermediate input cuts off its ancestors; inputs the outputs do not need are rejected; pass-through of an output that is an input; `compute` rejects a missing input and a value for a key that is not an input; snapshot behaviour; `warm` rejects a stage that holds a value depending on a parameter that another stage takes as input, and allows a stage that takes as input a value that another stage holds; `warm` computes shared work once, skips warm stages, and rejects stages that compute a shared key differently; concurrent calls compute the static part once; an expensive load before a cheap parameter (the shape of tuning in essapps); the default scheduler follows a replacement of `sciline.task_graph.DaskScheduler`; the `StreamProcessor` shape with a context update; `visualize_stages` fills what each stage computes with the color of that stage, and applies the styles of groups given by the caller.
- **Accumulators:** push order, the first push as result, reading without pushes, pushing combined values gives the same result.

### Nested drivers

Prototype drivers ran on fake workflows with the dependency structure of esssans (banks times sample and background runs) and Bifrost (triplets times runs, with a per-run step after combining the triplets), and gave the results of plain loops over `Pipeline.compute`.
Per-iteration work ran once per iteration of its loop, except work that depends on the bank alone (section 6.3).
A prototype `StreamProcessor` gave the results of `ess.reduce.streaming.StreamProcessor` for two dynamic keys in separate chunks, a context key, a context-only target, and `allow_bypass`.
These prototypes were built with an earlier helper, since dropped, which builds the same stages as section 5 for these shapes.
They need esssans and ess.reduce and are not part of this repository.
The `StreamProcessor` rewrite still has to pass the ess.reduce tests (section 10).

### LoKI multi-run reduction

`loki_validation.py`, next to this document, runs the esssans multi-run test workflow with one mask file, two sample runs, and two background runs.

- **Reference:** `with_pixel_mask_filenames`, `with_sample_runs`, and `with_background_runs`, using map/reduce.
- **Prototype:** `SansReduction` (section 6.1) with a contribute stage per run type and one finalize stage on the flat pipeline, the masks as a list parameter (section 6.4), and the detector IDs of the empty-beam run (section 6.1).

Results:

- `BackgroundSubtractedIofQ` and `BackgroundSubtractedIofQxy` are identical to the reference (`assert_identical`).
- Per-run `NormalizedQ` equals both a single-run computation and `compute_mapped`.
- Contributing, combining, and finalizing as separate calls, with the sample runs combined as a chain, gives the same result as `SansReduction.compute`.
- Provider call counts equal the reference, including a single read of the mask file, which requires `warm` over all three stages.
  The exception is `to_detector_mask`, called once instead of three times, because the masks depend on no run.
- Wall time with the naive scheduler: 6.9 s for the prototype and 7.1 to 7.2 s for the reference, in two runs.
  With sciline's default dask scheduler the reference is about 1.7 s faster, because the single graph computes the two sample runs in parallel threads; over stages this parallelism is up to the driver.
- Adding a second sample run after computing with one costs one contribution: one more `apply_pixel_masks` call and no second read of the mask file.
- Changing `QBins` means a new `SansReduction`, which makes the same provider calls as the first one.

Not validated: the rewrite of `StreamProcessor` against its real tests.

## 10. Migration

### What changes for each project

- **sciline:** adds `Stage`, `warm`, `Accumulator`, `Buffered`, `Reduced`, and `Pipeline.provide`, and later removes what section 7 lists.
  Users outside ESS need a documented replacement for `map(...).reduce(...)`: a loop over a stage with accumulators.
  No "experimental" label; the staged rollout below is the trial period.
  `Stage` also needs a `reporter` argument, so that progress reaches the ESS widgets.
- **ess.reduce:** the accumulators satisfy `sciline.Accumulator` once `maybe_hist` moves out of their base class, and a `Forwarder` is added.
  `StreamProcessor` is rewritten on stages against its existing tests (section 6.6).
  `parameter_mappers`, which maps a list-valued parameter to a `with_*` helper that returns a pipeline, is replaced by a common interface of the package objects (section 11).
  `get_parameters` is unaffected, because it works on the flat pipeline.
- **Reduction packages** (esssans, essreflectometry, essspectroscopy, essdiffraction, essnmx): the `with_*` helpers that fold are replaced by package objects, built from a configured pipeline, on which users set members and compute (section 6.1).
  Notebooks and tests change accordingly, and `visualize(compact=)` is dropped from notebooks.
  Specific points:
  - The pixel-mask folds in esssans and essdiffraction become a list parameter with providers (section 6.4).
  - `with_banks` in esssans, which maps without reducing, becomes a loop over a stage.
  - essreflectometry drops the `try/except` around each reduce: `Stage.dynamic_outputs` lists the keys that vary per run, and the driver pushes only those. Its `BatchProcessor` loses the fallback for mapped pipelines.
  - In bifrost, `NeXusData` depends on the run, so a bank fold inside a multi-run reduction sits inside per-run work.
    The package object uses the stages of section 6.3, with the triplets combined per run and the runs combined after.
    A list parameter with a provider that loops over the triplets would hide the loop inside a provider (section 8.6).
- **esslivedata:** the bifrost bank fold becomes a loop over a stage with an accumulator, computed before the `StreamProcessor` is built (section 6.5).
- **essapps:** the binding wraps the package objects and their stages instead of building its own (section 6.7).

### Findings of the survey that affect the migration

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
- esslivedata replaces `sciline.task_graph.DaskScheduler` to select its scheduler and checks at service startup that the replacement takes effect (section 8.7).
- Tracking: one issue in scipp/sciline for the sciline releases, one in scipp/ess for ess.reduce and the reduction packages.

### Order

1. sciline releases the additive part in a minor version, with the user guide on stages.
2. The packages migrate one at a time, each in its own pull request, while map/reduce still exists.
   esssans goes first, because it is the validated case and sets the pattern for package objects and for the interface that replaces `parameter_mappers`.
   The other packages depend only on the additive release and can then migrate in parallel.
   The `StreamProcessor` rewrite follows independently, but needs `Pipeline.provide` and the `reporter` argument of `Stage` in a sciline release first.
   It must keep honouring esslivedata's replacement of `sciline.task_graph.DaskScheduler`, or sciline first offers a public way to set the default scheduler and esslivedata switches to it.
   esslivedata follows the `StreamProcessor` rewrite through the essreduce version bump.
3. The last minor release of sciline deprecates `map`, `reduce`, `compute_mapped`, `get_mapped_node_names`, `constraints=`, and `visualize(compact=)`, once esslivedata has migrated.
4. Once no ESS package or esslivedata uses map/reduce, a major release of sciline removes it together with cyclebane, and the PEP 695 generics land.

## 11. Open questions

- **One package object per package, or one generic object?**
  The widgets need a common interface (set a parameter, set the members, compute).
  Either each package object implements it, or one generic object is built from a registry that maps member keys to accumulation keys; this is decided when the second package migrates.
- **Which run's detector IDs do the esssans background masks use?**
  `DetectorMasks` reads the detector IDs of the sample run set on the pipeline, and the background runs read it too (section 6.1).
  With several sample runs this is not defined, and `warm` rejects it (section 6.1).
  The validation script uses the empty-beam run, which gives identical results on the test data, where all runs have the same detector IDs.
  esssans decides when it migrates, for example detector IDs from the geometry or from a run chosen by a parameter.
- **Static work across processes.**
  Stages in one process share their static work through `warm`; a contribute call in a short-lived process recomputes it.
  This is the cost that essapps estimates for its stateless model of interactive work, not a new cost.
- **`sciline.v2` or a major release?**
  A `v2` namespace would let esslivedata and external users keep the old `Pipeline` next to the new one.
  But it bundles two independent changes (generics and map/reduce) under one name, invites mixing old and new pipelines in one process with obscure failures, and guarantees a second rename later.
  Recommendation: one major release, preceded by the additive minor release and the package migrations, in the order of section 10.
