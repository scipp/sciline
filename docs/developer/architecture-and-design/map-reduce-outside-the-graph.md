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

Sciline provides the stages, `enclose`, which puts stages inside a loop to build the stages of nested loops, and the `Accumulator` protocol with two accumulator factories, `Buffered` and `Reduced`.
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
| loop | One level of the driver's nested loops, given by the keys whose values the driver supplies on each iteration (the `inputs` of `enclose`). |
| outer stage | The stage that `enclose` returns for a loop. It computes, once per iteration, the values that the stages inside the loop read and that depend on the inputs of the loop. |
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
  A call must supply a value for each input.
  It must not supply a value for a key the stage uses but does not take as input, since the stage holds or computes that key itself and would ignore the value.
  Values for keys the stage does not use are ignored, so a driver can pass on everything that the stages of enclosing loops returned.
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
The scheduler default is the same as for `Pipeline` (section 8.8).

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

## 5. `enclose`

```python
bank_stage = Stage(pipeline, inputs=(Bank,), outputs=(Numerator, Denominator))
# bank_stage holds what it reads from the run, such as the content of the file
run_stage, bank_stage = enclose(pipeline, [bank_stage], inputs=(Filename,))
final_stage = Stage(pipeline, inputs=(Numerator, Denominator), outputs=(IofQ,))
# run_stage:   Filename -> what the bank stage held that depends on the run
# bank_stage:  those values and Bank -> Numerator, Denominator
# final_stage: Numerator, Denominator -> IofQ
```

A driver over several levels, such as banks within runs, needs to know which values the inner level reads from the outer one, so that it computes them once per iteration of the outer loop instead of once per iteration of the inner loop.
A stage built for the inner loop alone holds these values at its frontier.
`enclose` takes them from the frontier and returns one ordinary `Stage` for the outer loop, followed by the given stages, rebuilt.

### Behaviour

- **The frontier rule.**
  `enclose(pipeline, stages, inputs=...)` takes the frontier of the given stages.
  The values in it that depend on `inputs` are computed by the outer stage, once per iteration of the new loop, and the rebuilt stages take them as inputs.
  The values that do not depend on `inputs` stay held by the stages that read them.
- **From the inside out.**
  Nested loops are built from the innermost loop outwards: build the stages of the innermost loop, enclose them, then enclose the result.
  The order of the `enclose` calls gives the nesting; no stage names the loop that encloses it.
  A value stays held until the first `enclose` call whose inputs it depends on, so it is computed by the deepest loop whose inputs it depends on.
  It may skip levels:

  ```python
  bank_stage = Stage(pipeline, inputs=(Bank,), outputs=(Numerator, Denominator))
  run_stage, bank_stage = enclose(pipeline, [bank_stage], inputs=(Filename,))
  calibration_stage, run_stage, bank_stage = enclose(
      pipeline, [run_stage, bank_stage], inputs=(CalibrationFilename,)
  )
  # bank_stage reads Calibration, which depends on CalibrationFilename alone.
  # The first call leaves it held, the second moves it to calibration_stage.
  ```
- **Every stage inside the loop.**
  Each call gets every stage inside the new loop, not only the stages of the next inner loop.
  In the example above, the bank stage reads `Calibration` from the outermost loop directly.
  Left out of the second call, it would keep holding `Calibration`.
- **One call per loop.**
  All stages of one loop go into one `enclose` call.
  A stage returned by `enclose` takes the forwarded values as inputs.
  Enclosing it again in a loop over the same inputs does not forward them, and the driver fails with `Missing values for inputs [...]`.
  `enclose` cannot detect this.
  An input of a stage that depends on the inputs of the loop is either a forwarded value, such as the content of a file, or an accumulation key, such as `Numerator` for a per-run step after combining the banks; the graph does not tell them apart.
- **Work after combining.**
  A per-run step after the banks are combined is a stage from the accumulation keys of the bank loop.
  It is inside the run loop, so it goes into the same `enclose` call as the bank stage.
  A stage cuts its graph at its inputs, so the frontier of this stage does not contain what the accumulation keys are computed from.
  `enclose` forwards to it only the values it reads from the run, although in the pipeline the accumulation keys depend on the bank.
- **Outputs of the outer stage.**
  The outer stage outputs the keys given as `outputs`, followed by the forwarded values.
  `outputs` are keys the driver needs on each iteration of the new loop in addition to what the stages inside read, such as accumulation keys of the new loop.
  The driver pushes only those.
- **Declared outputs are checked, not derived.**
  An output of a given stage that depends on `inputs` but not on the inputs of that stage raises an error.
  In the rebuilt stage it would be a forwarded value passed through to the outputs, and the driver would push it once per iteration of a loop it does not vary in.
  The combined value would be wrong without any error: a key that depends on the run alone, pushed in the bank loop, is counted once per bank.
  An output that depends on the inputs of no loop is not rejected.
  The stage holds it, and it is not in `stage.dynamic_outputs`, so a driver that pushes only `dynamic_outputs` does not push it.
- **Snapshots.**
  `enclose` builds the outer stage and the rebuilt stages from `pipeline`, which must be the pipeline the given stages were built from.
  It raises an error if `pipeline` computes a key differently than when the given stages were built.
- **What the driver passes.**
  Each stage gets everything that the stages of the enclosing loops returned, `{**held_outer, **held_inner, Bank: b}`.
  `Stage.compute` ignores values for keys the stage does not use, so the driver does not select them.
  This also catches a stage left out of an `enclose` call: the stage still holds the forwarded value, and receiving it from the driver raises `The stage uses [...] but does not take them as inputs`.
- **Mistakes found when the stages are warmed or run.**
  `enclose` sees the stages of one loop, not the whole nesting.
  `warm`, given all stages of a driver, rejects a stage that holds a value depending on a parameter that another stage takes as input.
  This catches a stage left out of an `enclose` call, and a stage outside a loop that reads a value depending on the inputs of that loop (section 6.1).
  The stage would hold the value for the one member set on the pipeline, or fail if none is set.
  A stage enclosed twice over the same inputs fails in the driver, as above.
- **One level.**
  A single loop needs no `enclose`.
  A driver that builds a `Stage` directly pushes only `stage.dynamic_outputs`.
- **Drawing.**
  `visualize_stages(*stages)` draws the stages returned by `enclose` together, with a color per stage for what it computes per call, so that the derived boundaries can be checked by eye.

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

A driver that keys its accumulators by the dynamic outputs, as here, fails with a `KeyError` if an accumulator is missing.

### What is deliberately not in `enclose`

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
`enclose` is not the fix, because the background loop is not inside the sample loop; enclosed in it, each background run would be computed once per sample run.
The validation script takes the detector IDs from the empty-beam run instead, so the masks depend on no run and are computed once (section 11).

### 6.2 Several loops, one final stage

Sample runs and background runs are two loops on the same pipeline, and a final stage outside both loops takes the accumulation keys of both as inputs.
All stages are warmed together, so work they share, such as reading a mask file, is done once.

### 6.3 Runs times banks

Mapping over detector banks and runs at once (LoKI banks, Bifrost triplets) needs two levels.
A flat loop over (run, bank) pairs computes run-level work, such as loading monitors, once per bank, and pushes a run-level accumulation key once per bank; a loop per bank over the runs loads every run once per bank.
With `enclose`, the run stage computes the per-run values once per iteration of the run loop, and the bank stage takes them as inputs (section 5).
A run-level key declared as an output of the bank stage raises an error.

Bifrost merges the triplets of one run into one detector and concatenates the runs.
The per-run step after merging is a second stage inside the run loop, enclosed together with the triplet stage:

```python
triplet = Stage(pipeline, inputs=(NeXusDetectorName,), outputs=(EmptyDetector, NeXusData))
run_final = Stage(pipeline, inputs=(EmptyDetector, NeXusData), outputs=(NormalizedDetector,))
run, triplet, run_final = enclose(pipeline, [triplet, run_final], inputs=(Filename,))
final = Stage(pipeline, inputs=(NormalizedDetector,), outputs=(EnergyQDetector,))
```

The run stage outputs what both stages read from the run: `NeXusFile` and `PrimaryGraph` for the triplets, and the angles, monitor, and proton charge for `run_final`.
Each file is therefore opened once per run.
Enclosing the two stages in separate calls would give two run stages, and each would open the file.
Work that depends on the bank alone, such as LoKI's per-bank lookup table, is computed once per run and bank (section 8.7).
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
context_stage, *chunk_stages, finalize_stage = enclose(
    pipeline, [*chunk_stages, finalize_stage],
    inputs=context_keys, outputs=context_only_targets,
)
```

`set_context` calls the context stage and holds its result in a forwarder.
`accumulate` calls each chunk stage whose inputs the chunk supplies and pushes into the accumulators.
`finalize` calls the finalize stage.
What the chunk and finalize stages read from the context, today found by `_find_descendants` and `_find_parents`, is derived by `enclose`.
A target that depends on the context alone is an output of the context stage, given as `outputs`, and `allow_bypass` becomes an explicit input of the finalize stage.
A target that depends on no input belongs to no loop; a plain `Stage` without inputs computes it.
The accumulator classes, the grouping of accumulators by dynamic keys, the validation of key sets, `on_finalize`, `clear`, and `visualize` stay.
The module shrinks to its policy, which values are transient and which are held, as scipp/ess#732 proposed.

With one context stage, an update of any context key recomputes all context-derived values, where `StreamProcessor` today recomputes only what depends on the updated keys.
In esslivedata this costs compute and changes no results.
The only reset tied to the context, the reset-on-move of `NoCopyAccumulator`, compares coordinate values on each push, and values recomputed from unchanged inputs are identical; nothing inspects which context nodes were recomputed.
The extra work is at most one rebuild of the detector projection when a rare key (an ROI edit, or a wavelength lookup table after a chopper change) is updated in a job with moving detector geometry.
The calls inside `StreamProcessor` must also honour esslivedata's choice of scheduler (section 8.8).

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
`enclose` derives the same boundaries but returns one plain stage per loop, so every held value is an object the driver can inspect, clear, serialize, or send to another process: the explicit lifetime that scipp/sciline#241 asks for.
`Stage` holds the values at its frontier itself.
A stage without inputs followed by a forwarder would do the same.
It is built in because every stage needs it.

### 8.2 No generic network object (deferred)

A general object could hold stages and connectors and route pushes through them.
The real uses disagree on exactly this routing: a chunk runs immediately because it cannot be kept, a context update runs immediately but must not touch the accumulators, a loop over runs can run lazily, `clear` discards accumulators but keeps the context, and rolling windows and `on_finalize` add their own rules.
A generic object would either expose parameters for all of this or hide one choice, and it would be a graph of stages with its own scheduler, which is nested workflows one level up (section 8.9).
Each real use is a loop of under twenty lines over plain objects, and that loop is where the policy belongs.
If package objects turn out to repeat the same code, that repetition is the basis for a generalization of the package objects; if essapps phase 3 needs connectors on process boundaries, that is a placement layer over the same stages and connectors.

### 8.3 Boundaries between levels are derived

Three callers need the same boundary: the context frontier of `StreamProcessor`, runs times banks in esssans, and runs times triplets in Bifrost.
Chosen by hand, it fails silently: a run-level key left out is recomputed per bank, which costs time but gives the right result.
`enclose` derives it from the frontier of the stages inside the loop (section 5).

Stages are connected in two ways:

| Connection | From | To | Given by |
|---|---|---|---|
| forwarded value | outer stage | stages inside its loop | derived by `enclose` from the frontier |
| accumulation key | stages inside a loop | a stage after the loop | declared as outputs of one stage and inputs of the next |

The accumulation keys are declared, not derived: the combine happens outside the graph, so the graph cannot tell that a key such as `NormalizedDetector` comes after combining the triplets.
An output declared on the stage of the wrong loop raises an error instead of being moved to the loop it varies in, where the driver may have no loop or accumulator for it.

The nesting of the loops is given by the order of the `enclose` calls, from the inside out.
An alternative, `split(pipeline, *parts)`, took all loops at once, one `Part` per loop with its inputs, its outputs, and a `parent`, the part of the enclosing loop.
For three loops with a per-run step after combining, `split` and `enclose` build identical stages.
`split` was dropped because `parent` declared the forwarded values only indirectly, which made the API hard to understand.
`enclose` builds on the frontier, which users of `Stage` already know: a held value that depends on the inputs of the new loop is forwarded.

The cost of `enclose` is that some mistakes are found when the stages run, not when they are built.
`split` saw all loops at once and rejected, at construction, a read from a loop that does not enclose the reader and a part whose enclosing part was not given.
`enclose` sees one loop at a time, so `warm` takes over these checks: it sees all stages of a driver and rejects a stage that holds a value depending on a parameter that another stage takes as input (section 5).
This relies on the driver warming its stages together, which it does anyway so that shared work is done once.
A stage enclosed twice over the same inputs is found only in the driver.

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

### 8.7 Loops nest as a tree

Each `enclose` call puts stages inside one loop, so each loop is inside at most one other.
Two cases would need a stage inside two loops that do not nest:

- Work that depends on the bank alone, held across all runs; with runs as the outer loop, it is computed once per run and bank.
- Context keys updated independently in `StreamProcessor`; with one context stage, any update recomputes all context-derived values (section 6.6).

Both cost compute only, and little: the per-bank lookup table in LoKI is cheap, and the extra context work in esslivedata is at most one rebuild of the detector projection on a rare update.
Two enclosing loops would make the owner of a value ambiguous when it depends on both, and holding values across all iterations of another loop is a memory decision that belongs to the driver.

### 8.8 Default scheduler

`scheduler_or_default` lives in `sciline.task_graph` and looks up `DaskScheduler` in that module on each call, for pipelines and stages alike.
esslivedata replaces `sciline.task_graph.DaskScheduler` to select its scheduler, including for the calls inside `StreamProcessor` that it cannot pass a scheduler to; a sciline test checks that this changes the default of both.
A public way to set the default scheduler would be cleaner.
Until it exists, sciline keeps the replacement working for stages (section 10).

### 8.9 Not nested workflows

Nested workflows, sub-workflows with their own parameters inside a node of an outer workflow, were rejected earlier for plumbing (parameters passed across boundaries) and opacity (a boundary hid what was inside).
Here the author writes one flat graph, and stages are cut from it by the driver, from the keys the driver names.
Every parameter is set on the flat pipeline and reaches every stage, and composition across stages is ordinary Python, so no graph contains another graph.

### 8.10 Names

- **Accumulator**, **contribution**, and *contribute*/*combine*, as in Beam, Flink, and Spark, and in essapps.
  `fold` was avoided because it means reshaping in scipp (essapps uses *fold* for a long-lived runner, not for the operation).
- **Accumulation key**, matching the essapps term.
- `Forwarder` is the working name in ess.reduce; the final name is ess.reduce's decision.

## 9. Validation

### Unit tests

The implementation is in `src/sciline/stage.py` and `src/sciline/accumulators.py`, tested in `tests/stage_test.py`, `tests/enclose_test.py`, and `tests/accumulators_test.py`.
It runs on sciline `main`.
An earlier version of the stage tests was run on the generics branch and passed; `enclose` has not been run there.

The tests cover:

- **Stage:** static part computed once and dynamic part per call; an intermediate input cuts off its ancestors; inputs the outputs do not need are rejected; pass-through of an output that is an input; `compute` rejects a missing input and a value for a key the stage uses but does not take as input, and ignores a value for a key it does not use; snapshot behaviour; `warm` rejects a stage that holds a value depending on a parameter that another stage takes as input, and allows a stage that takes as input a value that another stage holds; `warm` computes shared work once, skips warm stages, and rejects stages that compute a shared key differently; concurrent calls compute the static part once; an expensive load before a cheap parameter (the shape of tuning in essapps); the default scheduler follows a replacement of `sciline.task_graph.DaskScheduler`; the `StreamProcessor` shape with a context update; `visualize_stages` applies the styles of groups given by the caller.
- **enclose:** three nested loops give the result of plain loops over `Pipeline.compute`; a value is computed by the innermost loop whose inputs it depends on, also skipping a loop; a loop inside a stage whose input is an accumulation key reads from that stage; a value that depends on no loop stays held; per-iteration work runs once per iteration of its loop; `outputs` of the outer stage that an inner stage reads are output once; the outer stage uses the given scheduler and the inner stages keep theirs; a stage left out of a loop is rejected by `warm`, and rejects what the loop computes when the driver passes it on; an output that does not vary with the inputs of its stage, a loop that no stage reads from, a pipeline changed since the stages were built, unknown outputs, and unneeded inputs are rejected; `visualize_stages` fills what each stage computes with the color of that stage.
  The user guide on stages (`docs/user-guide/stages.ipynb`) runs two and three nested loops and a per-file step after combining the banks, and shows that each file is read once and each calibration is loaded once.
- **Accumulators:** push order, the first push as result, reading without pushes, pushing combined values gives the same result.

### Nested drivers

Prototype drivers with `split` (section 8.3) ran on fake workflows with the dependency structure of esssans (banks times sample and background runs) and Bifrost (triplets times runs, with a per-run step after combining the triplets), and gave the results of plain loops over `Pipeline.compute`.
Per-iteration work ran once per iteration of its loop, except work that depends on the bank alone (section 8.7).
A `StreamProcessor` on `split` gave the results of `ess.reduce.streaming.StreamProcessor` for two dynamic keys in separate chunks, a context key, a context-only target, and `allow_bypass`.
These prototypes need esssans and ess.reduce and are not part of this repository.
They were not run with `enclose`; the esssans shape is validated with `enclose` on real data below.
They have at most two nested loops and a step after combining; for three such loops, `enclose` builds the same stages as `split`.
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

### LoKI runs times banks

`loki_banks_validation.py`, next to this document, runs the esssans LoKI workflow over two sample runs and the nine detector banks of `loki_coda_file`, the only multi-bank test file.
The second run is a copy of that file with the detector event IDs reversed and the monitor event time offsets scaled, so that both per-run and per-bank values differ between the runs.

- **Reference:** `with_banks(with_sample_runs(...))`, using map/reduce, which gives `IntensityQ[SampleRun]` per bank, combined over the runs.
- **Prototype:** a bank stage from `NeXusDetectorName` to `NormalizedQ[SampleRun, Numerator]` and `NormalizedQ[SampleRun, Denominator]`, enclosed in a loop over `Filename[SampleRun]`, and a final stage to `IntensityQ[SampleRun]`, with one set of accumulators per bank.

Results:

- `enclose` and `warm` accept the stages with no keys added by hand.
  The run stage outputs `NeXusFileSpec`, `ElasticCoordTransformGraph`, `MonitorTerm`, and the source and sample positions of the sample run.
- `IntensityQ[SampleRun]` is identical to the reference for every bank (`assert_identical`).
  The script checks that the results are not all NaN: with the 200 wavelength bins of the esssans user guide, the 5 pulses in the file leave wavelength bins without monitor counts, and every value of I(Q) is NaN. It uses 20 bins.
- The monitor term is computed once per run and the detector data are assembled once per run and bank, as in the reference.
- Wall time with the naive scheduler: 3.7 to 3.8 s for the prototype and 4.7 to 4.8 s for the reference, in two runs.

Not validated: the rewrite of `StreamProcessor` against its real tests.

## 10. Migration

### What changes for each project

- **sciline:** adds `Stage`, `warm`, `enclose`, `Accumulator`, `Buffered`, `Reduced`, and `Pipeline.provide`, and later removes what section 7 lists.
  `enclose` belongs in sciline because it needs the dependency graph of the pipeline, and users outside ESS need a documented replacement for `map(...).reduce(...)`: a loop over a stage with accumulators, and `enclose` for nested loops.
  No "experimental" label; the staged rollout below is the trial period.
  `Stage` also needs a `reporter` argument, so that progress reaches the ESS widgets.
- **ess.reduce:** the accumulators satisfy `sciline.Accumulator` once `maybe_hist` moves out of their base class, and a `Forwarder` is added.
  `StreamProcessor` is rewritten on `enclose` against its existing tests (section 6.6).
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
- esslivedata replaces `sciline.task_graph.DaskScheduler` to select its scheduler and checks at service startup that the replacement takes effect (section 8.8).
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
