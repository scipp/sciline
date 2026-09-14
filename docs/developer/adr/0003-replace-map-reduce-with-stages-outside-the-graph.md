# ADR 0003: Replace map/reduce with stages composed outside the graph

- Status: proposed
- Deciders: Simon (proposer); to be discussed with Jan-Lukas, Johannes, Mridul, Neil, Sunyoung
- Date: 2026-09-11

Design document with prototype and validation: [Map and reduce outside the graph](../architecture-and-design/map-reduce-outside-the-graph.md).

## Context

`Pipeline.map` and `Pipeline.reduce` put a loop over members and a combine step inside the task graph.
This has been implemented twice, as parameter tables (23.08) and on cyclebane (24.06).
Sciline documents it for all users in the parameter-tables guide, so whatever replaces it has to be usable without the ESS packages.

In the PEP 695 generics prototypes (scipp/sciline#236 and #237, both closed unmerged) every breaking change and every open question came from map/reduce.
Dependents of a mapped generic key do not exist until a target is named, so `map` needed a forward pass that instantiates rules to a fixed point, restricted to keys derived from the mapped ones to avoid spurious sinks.
`reduce` without an explicit key resolves the unique sink at call time, but under rules the sink set grows as instantiation proceeds.
Requiring unique node names forbids the `pipeline[C] = pipeline[C].map(...).reduce(...)` idiom every ESS package uses.
And the branch depended on an unreleased cyclebane (scipp/cyclebane#32, closed unmerged).
Users cannot write a mapped key; they need `get_mapped_node_names` and `compute_mapped`, which depend on pandas and on cyclebane's node-name classes.
`groupby` was announced in the guide and never implemented; `index_names` and `indices` have no users in the ESS packages.

At the same time, three consumers need an operation map/reduce cannot express: run one part of a graph repeatedly with some values supplied per call, combine the results outside, run the rest once.

- `ess.reduce.streaming` (1080 lines) builds this partition by hand, pruning branches by assigning `None`, grafting subgraphs, and patching provider annotations at runtime; scipp/sciline#241 records the requirements: no graph rebuild per value, explicit lifetime of supplied values, culling of a supplied key's ancestors.
- The planned reduction-service architecture (essapps) needs the same partition three times: caching everything upstream of a cheap parameter, splitting a reduction over members into contribute, combine, and finalize so that members can be processed in separate processes, and running two halves of one graph in two processes.
- esssans, essreflectometry, and bifrost fold runs at one or more keys in their workflow modules and graft the result back onto the pipeline (`with_sample_runs` and siblings); essdiffraction and essnmx do so in notebooks and tests.
  `ess.reduce.parameter_mappers` and the parameter widgets depend on the grafted pipeline.

## Decision

Remove `map`, `reduce`, `index_names`, `indices`, `get_mapped_node_names`, `compute_mapped`, `visualize(compact=)`, and the cyclebane dependency from sciline; drop the planned `groupby`.
`constraints=` goes in the same release: the PEP 695 generics match rules structurally and do not need it.
The graph partition becomes the primitive:

1. **`Stage`, `warm`, and `provide`, in sciline.**
   A stage is the part of a pipeline from named input keys to named output keys.
   Its static part, everything the outputs need that does not depend on the inputs, is computed once and held at its frontier: the static values the dynamic part reads, plus static outputs.
   A call supplies exactly the inputs, computes only the dynamic part, and releases supplied and computed values on return.
   An input may be a parameter or an intermediate node; an output that is an input is passed through.
   A stage is built from a copy of the concrete task graph and is unaffected by later changes to the pipeline; parameter values are captured by reference.
   `stage.keys` and `stage.frontier` expose the partition.
   `warm(*stages)` computes the static parts of several stages of one pipeline in one run, so shared static work is done once.
   `provide(key, callable)` registers a provider for a key known only at runtime, as scipp/sciline#241 asks.

2. **`Accumulator`, `Buffered`, `Reduced`, `Aggregation`, and `compute_members`, in sciline.**
   An accumulator is any object with `push(value)` and a `value` property: values go in, the combined value comes out.
   Whether it buffers or keeps a running result is its own choice.
   `Buffered(func)` is a factory for one that holds every pushed value and applies an n-ary function on `value`, which is how every existing combine function is used.
   `Reduced(func)` is a factory for one that holds only a running result of an associative binary function, for sums of large arrays.
   Member keys are the keys supplied per repetition, such as `Filename[SampleRun]`; one row of values for them is a member.
   An accumulation key is a key at which per-member values are combined; a contribution is the dict of values at the accumulation keys for one member.
   An aggregation is two stages of one flat pipeline, contribute (member keys to accumulation keys) and finalize (accumulation keys to outputs, optional), with an accumulator factory per accumulation key between them.
   It holds no contributions and no member table; its stages hold their frontiers, and it makes fresh accumulators for every combine.
   Parameters are set on the pipeline before the aggregation is built; a changed parameter means a new aggregation.
   `agg.accumulation_keys` lists the requested keys that depend on the members; one that does not is computed by finalize instead.
   `contribute`, `combine`, and `finalize` are separate entry points, so the three steps can run in different processes with the contribution serialized between them; `compute(table)` is the loop for a caller that holds nothing, and it pushes each contribution as it is made.
   Parallelism over members is the caller's: `contribute` is a plain call to map over rows with threads, processes, or dask, pushing results as they arrive.
   A member table is a `Mapping[label, Mapping[Key, value]]`; pandas is not involved.
   `compute_members(pipeline, members=, key=, table=)` computes one key per member and replaces `compute_mapped`.

3. **Connectors, in ess.reduce.**
   The objects between stages of a stream: the accumulators of `ess.reduce.streaming`, which satisfy sciline's `Accumulator` protocol as they are once `maybe_hist` moves out of their base class, and a `Forwarder`, which holds a context until it is replaced.
   `clear` is the convention for a reused accumulator and stays in ess.reduce with the drivers that reuse them.
   No shared base class beyond the protocol.
   Every held value is a plain object the caller can inspect, clear, or serialize.

4. **Drivers and package objects own topology, state, and policy.**
   A driver is the loop that calls stages and pushes into connectors; a package object is what a reduction package returns to users in place of a map/reduced pipeline.
   `StreamProcessor` becomes a driver over three stages, a forwarder, and accumulators.
   esssans's object holds the pipeline, one aggregation per run type, the finalize stage over the accumulation keys of both, and the contributions by filename; on a parameter change it rebuilds an aggregation if `key in agg.contribute_stage.keys` and the finalize stage if that reads the key.
   Hierarchy (banks over runs, angle groups inside a run) is composition of such objects; nothing nests inside a graph.

A drop-in replacement for the map/reduced pipeline is a non-goal.

## Alternatives considered

- **Keep map/reduce and land the PEP 695 generics on top.**
  The four problems listed under Context did not close in two prototype passes.
- **Keep map/reduce, deprecated, and add `Stage` and `Aggregation` beside it.**
  The additive part is possible: the prototype runs on sciline `main`.
  But map/reduce keeps cyclebane and blocks the generics, and two mechanisms for one job would have to be maintained side by side.
  The rollout is staged instead: stages and aggregations first, packages migrated one at a time, removal last.
  Users who depend on map/reduce and do not need the new generics can pin the last release before removal; keeping the old `Pipeline` in a separate namespace would serve them equally, but nobody intends to maintain it, so pinning is the honest offer.
- **A bridge back into the pipeline** (`as_pipeline()`, synthesizing providers for the accumulation keys) so that `with_*` helpers keep returning a pipeline.
  Prototyped and dropped: it was the only way two aggregations composed, so it was the mechanism rather than a bridge; it fixed the member table at construction; and it hid a loop in a provider with no per-member errors, progress, or visualization.
- **A `Stage` with tiers of inputs and a hold policy per tier**, deriving all boundaries itself.
  Same information as connectors, held inside one class and putting policy into sciline.
- **An aggregation object that sets parameters and holds state** (member tables with groups, contributions kept by a row-equality heuristic, a held per-member frontier, invalidation rules in `__setitem__`).
  Prototyped and stripped: it merged `Pipeline`'s job with orchestration, reproduced the map/reduced pipeline, and was hard to reason about; on LoKI the held frontier bought nothing because the wavelength conversion reads a parameter.
- **An n-ary combine function per accumulation key**, as `reduce(func=)` takes today.
  It forces the aggregation to hold every contribution before combining, so anyone who cares about memory writes their own loop.
  With an accumulator the buffer-versus-running choice sits on the object the author passes; `Buffered` keeps the n-ary function as the thing authors write, and `Reduced` covers the running case without a hand-written class.
- **A generic network object** holding stages and connectors and routing pushes.
  The real shapes disagree on the push policy (eager chunks, context updates that keep accumulators, lazy table aggregations, `clear` semantics), and it is the nested-workflow shape one level up.
  Deferred: if the package objects turn out to repeat the same thirty lines, that object is their generalisation.
- **Nested workflows**, a graph inside a node.
  Rejected earlier for the plumbing between levels and the opacity of the inner graph; stages are derived from one flat graph and add nothing to it.

## Consequences

### Positive

- Sciline loses map/reduce and cyclebane; the PEP 695 generics land without their open questions.
  16 tests that use map/reduce go; the branch's map-related tests are trimmed.
- Sciline keeps a documented replacement: one `Aggregation` where the guide had `map(...).reduce(...)`.
  The parameter-tables guide becomes a guide on stages and aggregations, with an example of contribute mapped over rows in parallel.
  Nothing ships as experimental: the additive minor release, with esssans migrated onto it, is the trial period before removal.
- `StreamProcessor` is expected to shrink to its policy, meeting scipp/sciline#241 by construction; the rewrite is not done.
- One vocabulary across sciline, ess.reduce, and essapps: stage, accumulator, accumulation key, contribution, contribute/combine/finalize.
  The stateful object is an accumulator as in Beam, Flink, and Spark; the three-phase shape is aggregation as in those and pandas; `fold` would clash with scipp's reshaping `fold`.
- Validated on the LoKI multi-run reduction: results identical to the map/reduce reference and equal provider call counts.
  6.9 s against 7.2 s with both on the naive scheduler; under sciline's default dask scheduler the reference is about 1.7 s faster because the one graph runs the two sample runs in threads (see below).
  Adding a run costs one contribution.
- Every parameter is set on the flat pipeline and reaches every stage; what is held is visible as objects.

### Negative

- Breaking for the `with_*` callers in esssans, essreflectometry, and bifrost, for `parameter_mappers`, for essreflectometry's `BatchProcessor`, for notebooks, and for esslivedata's bifrost bank fold.
  Packages migrate one at a time while map/reduce still exists; esslivedata pins sciline until it follows.
- The GUI needs one protocol across package objects; whether that is a base class or one generic object from a registry of member key to accumulation key is decided when the second package is migrated.
- Parallelism over members, which one graph under the dask scheduler gave for free, becomes the caller's responsibility.
- A map/reduce whose reduced key sits inside another's per-member work (esssans pixel masks) is not an aggregation; it becomes a list parameter and a provider.
- A requested accumulation key that does not depend on the members is computed by finalize without error; `accumulation_keys` reports it, and a consumer that declares its accumulation keys, a package test or the essapps binding, compares the two.
- Stages hold their frontier for their lifetime, and package objects hold contributions per member, which are binned events under `ReturnEvents=True`; package objects need `clear`.
- Each aggregation walks the graph at construction and again on every parameter change that reaches it.
- Visualization of a mapped graph goes with `compact=`; a visualization over stages and connectors replaces it.
- Progress reporting has to be threaded through `Stage`; the prototype has none.
- networkx, so far a transitive dependency through cyclebane, becomes a direct one.
- Not yet done: the `StreamProcessor` rewrite against its streaming and visualize tests; the choice between `sciline.v2` and a major release (recommendation: major release).
