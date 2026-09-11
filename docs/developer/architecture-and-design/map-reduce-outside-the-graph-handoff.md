# Handoff: reviewing and rolling out stages outside the graph

Written 2026-09-11 at the end of the design session, for a new session that reviews the proposal in detail and plans staging and rollout across sciline, ess.reduce, the reduction packages, esslivedata, and essapps.
Working document; drop before the branch merges.

## Read first

1. [ADR 0003](../adr/0003-replace-map-reduce-with-stages-outside-the-graph.md), proposed: the decision, alternatives tried, consequences. One page.
2. [Map and reduce outside the graph](map-reduce-outside-the-graph.md): the design with semantics, evidence, migration, open questions.
3. `stage-prototype/stage.py` (about 250 lines): `Stage`, `warm`, `Forwarder`, `Fold`, `compute_members`. The docstring at the top is the shortest statement of the design.
4. `stage-prototype/loki_validation.py`: `SansReduction`, the package object esssans would return, about thirty lines, and the validation against the map/reduce reference.
5. essapps branch `architecture-sketch`, `docs/developer/stages.md`: how the pieces map onto D8, D13, D14, D15, and phase 3.

## Where things are

| What | Where |
|---|---|
| Design branch | sciline `map-reduce-outside-the-graph`, off `main`; commits `84e44d1` (first pass), `ddd7012` (connectors), `0bdbaa3` (stateless Fold), `77f718f` (ADR) |
| Prototype and tests | `docs/developer/architecture-and-design/stage-prototype/`; 21 tests |
| essapps companion note | branch `architecture-sketch`, `docs/developer/stages.md`, commits `a5ccdb8`, `05a1ba7`, `8c77e38` |
| Environment | `/opt/conda/bin/python` has sciline editable from `/workspace/sciline/src` and `ess.sans` installed; no venv in the repo |
| Run tests | `cd docs/developer/architecture-and-design/stage-prototype && python -m pytest -q stage_test.py stream_test.py` |
| Run validation | same directory, `python loki_validation.py`; about 40 s, downloads LoKI tutorial data on first run |
| ESS monorepo checkout | `/workspace/ess/packages/{essreduce,esssans,essreflectometry,essspectroscopy,essnmx,essdiffraction,...}`; esslivedata at `/workspace/esslivedata` |
| Stale copy | `/workspace/sciline/.scratch/proto/` is the first-pass prototype; delete |

## The design in ten lines

- `Stage(pipeline, outputs, inputs)`: the graph from input keys to output keys; static part held at its frontier, dynamic part per call; snapshot of the pipeline. `warm(*stages)` computes several static parts in one run. Sciline.
- Connectors between stages: `push`, `value`, `clear`. Reducers (accumulators, n-ary combine), `Forwarder` (held context), a dict by member label (held contributions). ess.reduce.
- `Fold(pipeline, members, at, outputs=())`: contribute stage, optional finalize stage, combine per cut key; holds nothing; `contribute`/`combine`/`finalize`, `compute(table)`. Sciline (decided 2026-09-11; mechanism, not policy).
- Package objects own topology, state, and policy: pipeline, folds, shared finalize stage, contributions, and the rule for what a parameter change keeps (`key in fold.contribute_stage.keys`). `StreamProcessor` is the same kind of object for streams.
- Non-goals: a drop-in replacement for the map/reduced pipeline; a generic network object; a boundary builder. Reasons in the ADR.

## How the design got here

Three passes in two days; knowing why each was rejected matters for review.

1. `Stage` + `Fold` with `as_pipeline()`, a bridge synthesizing providers so `with_sample_runs` could keep returning a pipeline. Rejected: the bridge was the only way folds composed, fixed the member table, hid a loop in a provider.
2. Connectors between stages, `Fold` as a driver with groups, settable members and parameters, contributions held by a row-equality heuristic, a held per-member frontier. Rejected by Simon: merges `Pipeline`'s job with orchestration, models esssans's map/reduced pipeline, hard to reason about. The held frontier bought nothing on LoKI because the wavelength conversion reads a parameter.
3. Stateless `Fold`, state in package objects. Current.

Simon's direction, verbatim: "Set pipeline params on the pipeline. Fold manages the high-level pipeline (stage) orchestration only."
A tiered `Stage` (input groups with hold policies inside the class) was proposed and rejected in favour of connectors: same information, hidden in one class, policy in sciline.

## Open questions, with recommendations

- **Combine protocol.** n-ary functions in `Fold`, accumulator objects in `StreamProcessor`. Recommendation: keep both for now; consider `Fold` accepting an accumulator per cut key once `StreamProcessor` is rewritten and the accumulator base is cleaned up.
- **Member tables.** `compute`/`compute_members` accept a dict of columns or a DataFrame. Recommendation: `Mapping[label, Mapping[Key, value]]` only; pandas stays out.
- **One package object or many.** Decides whether the GUI protocol (set parameter, set members for a member key, compute) is a base class, a protocol, or one generic object from a registry of member key to cut. Recommendation: write esssans's by hand, then essreflectometry's, then decide.
- **`sciline.v2` namespace or major release.** Recommendation in the doc: major release; ESS monorepo migrated in one PR; esslivedata pins.
- **Parallel members.** Lost the free dask parallelism of one graph (about 1.7 s of 7 s on LoKI). An executor on `Fold.compute`, or the caller's dask delayed.
- **Names.** stage, cut, contribution, contribute/combine/finalize, Forwarder; essapps says accumulation point, stage output. Settle before code lands.

## Rollout: a staging to evaluate

The key property: `Stage` and `warm` work on sciline `main` today (the prototype does), so the additive part can land before the breaking part.

1. **sciline, additive.** `Stage`, `warm`, `provide(key, callable)`; building a task graph with missing values (already what `HandleAsComputeTimeException` does). Replace the prototype's private access (`TaskGraph._graph`, `Provider.parameter`, `ArgSpec`) with public construction. Docs: a guide on stages. No breaking change.
2. **ess.reduce, additive.** `Fold`, `compute_members`, `Forwarder`; move `maybe_hist` default out of `Accumulator.push`; rewrite `StreamProcessor` on `Stage` against its 35 tests (`streaming.py`, 1081 lines; `_FedWorkflow`, `_build_streaming_workflow`, `_find_descendants`, `_find_parents`, `_map_context_to_cached_nodes`, `allow_bypass`, `visualize` classification). map/reduce still present, nothing else changes.
3. **Packages, one at a time, each its own PR.** Replace the `with_*` helpers by a package object; migrate notebooks and tests. Sites, from the survey in the design doc:
   - esssans: `_set_runs`/`with_sample_runs`/`with_background_runs` (4 cut keys × 2 run types, `merge_contributions`); `with_pixel_mask_filenames` becomes a `PixelMaskFilenames` list parameter and two providers (`DetectorMasks` reads `DetectorIDs` from `EmptyDetector[SampleRun]`, so the mask fold is inside per-run work); `with_banks` becomes a loop; `parameter_mappers` registrations in `workflow.py` lines 144-150; `beam_center_from_center_of_mass` callers use `with_pixel_mask_filenames`.
   - essreflectometry: up to 7 cut keys, concat and `_any_value`, `try/except` around reduces that becomes the static-cut rule.
   - essspectroscopy (bifrost): two cut keys at different depths; the esslivedata `EmptyDetector` bank fold.
   - essnmx: panels and MTZ files, notebooks use `Fold` directly.
   - essdiffraction (DREAM): two-column member table; `with_pixel_mask_filenames` as in esssans.
   - isissans: zoom monitors, concat along a new dim with an assert-unique position combine.
   - ess.reduce `BatchProcessor` (mapped/unmapped fallback) and `workflow.py` `assign_parameter_values` (`parameter_mappers`).
   Re-survey the sites before planning; the table in the design doc is from 2026-09-10 and greps of `.map(` and `.reduce(` in `/workspace/ess` and `/workspace/esslivedata`.
4. **esslivedata.** Bifrost bank fold becomes `Fold(...).compute(...)` set as a parameter before the `StreamProcessor` is built; consumer of the rewritten `StreamProcessor`. Pins sciline until migrated.
5. **sciline, breaking.** Remove `map`, `reduce`, `groupby`, `index_names`, `indices`, `get_mapped_node_names`, `compute_mapped`, `compute(pandas.Series)`, `visualize(compact=)`, `constraints=`, cyclebane. Land the PEP 695 generics from branch `235-pep695-single-model-prototype` without the map/reduce parts. Last minor release before it deprecates `map`, `reduce`, `constraints=`.
6. **essapps.** Apply the suggested edits listed at the end of `stages.md` (D8, D13, D14, D15, glossary); the D3/D6 spike's fake workflow with two accumulation points is a `Fold`.

Whether step 5 can wait for step 3 to finish everywhere, or whether the PEP 695 work forces it earlier, is the main scheduling question.

## Things to verify in review

- The prototype runs on `main` and its generics test uses `sciline.Scope`; on the PEP 695 branch all but the generics test passed at the first pass, not rerun since.
- `Stage` needs the concrete task graph with generics instantiated; in v2 the pipeline holds it directly. Check how `underlying_graph`/`output_keys()` under demand-driven generics interact with `_task_graph`.
- The single-sink trick in `_compute` keeps a key alive when another requested key consumes it; check the dask scheduler path.
- `Stage` is a snapshot because `pipeline.get()` builds a fresh task graph; confirm this holds for values set through `__setitem__` after grafting a sub-pipeline.
- `Fold` builds the probe stage and the contribute stage separately (two graph walks); fine for a prototype, tidy in the real thing.
- Memory: package objects hold contributions by label; for `ReturnEvents=True` these are binned events. The essapps note says contributions in a session, nothing else; the notebook object should expose `clear`.
- The mask change in esssans alters call structure (masks built per run from the run's IDs) but not results; identical on LoKI. Check esspowder/DREAM does the same.

## Not done

- `StreamProcessor` rewrite against its real tests.
- Parallel members.
- Rerun of the prototype on the PEP 695 branch.
- Any code in the ESS packages.
