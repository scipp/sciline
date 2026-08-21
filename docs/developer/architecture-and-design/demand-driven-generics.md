# Demand-driven generics

```{note}
Living document accompanying the discussion of
[#235](https://github.com/scipp/sciline/issues/235) and the prototypes
[#236](https://github.com/scipp/sciline/pull/236) and
[#237](https://github.com/scipp/sciline/pull/237).
It records context, the mechanism, measured evidence, and the open design
questions. Decisions, once made, should graduate into ADRs and this document
should be updated or retired.
```

## Problem statement

Sciline's premise is that type annotations are the wiring of a workflow:
a provider's parameter and return annotations are dependency-injection keys.
Generic providers extend this to families of keys, typically parameterized by
a run type:

```python
def clean(data: RawData[Run]) -> CleanData[Run]: ...
```

On `main`, generic providers are expanded *eagerly* at insertion: every type
variable must carry constraints (`Run = TypeVar('Run', SampleRun, BackgroundRun)`),
and `Pipeline.insert` instantiates the provider for the full cross-product of
all its type variables' constraints. This has three consequences:

- The graph contains all instantiations whether needed or not
  (`visualize(cluster_generics=...)` exists to tame the resulting clutter).
- Constraints must be enumerable at class-definition time, which technique
  packages cannot do for downstream run types. The `constraints=` argument of
  `Pipeline` was added to patch this.
- The mechanism identifies type variables *by object identity* across
  providers and the constraints mapping.

Python 3.12 (PEP 695) breaks the third point fundamentally: `class Raw[Run]`
and `def foo[Run](...)` each mint their own scoped `Run` that compares unequal
to every other `Run`, and scoped type variables have empty `__constraints__`
unless spelled out per definition. There is no cross-definition identity to
key a constraints mapping on. Separately,
[#233](https://github.com/scipp/sciline/issues/233) shows `sl.Scope` breaking
on newer Python via metaclass conflicts; plain PEP 695 generics would fix
that, but only if sciline can consume them.

### The enabling observation

Sciline needs cross-definition type-variable identity in exactly one place:
the constraints lookup. Everything else — binding a provider's type variables
and substituting them into its argument and return annotations — is
*per-provider*, and within a single provider PEP 695 scoped type variables are
self-consistent: all annotations of `def foo[Run](x: Raw[Run]) -> Processed[Run]`
share one `Run` object (verified on CPython 3.12). So if instantiation is
driven by *demand* instead of by declared constraints, the identity problem
disappears, and with it the constraints mechanism.

## The mechanism (as prototyped in #236/#237)

A pipeline is a set of **facts** and **rules**:

- *Facts*: concrete providers and concrete params. They live in the cyclebane
  graph exactly as on `main`. The graph never contains a generic node.
- *Rules* ("templates"): generic providers and values set for generic keys.
  They live in one insertion-ordered list outside the graph.

Rules are instantiated by structural unification of type patterns with
concrete keys, in two directions:

**Backward chaining** (the core): when a concrete key is demanded (`get`,
`compute`, `__getitem__`, or transitively as a dependency), and no fact
satisfies it, the latest-registered rule whose return pattern unifies with the
key is instantiated: `Processed[A]` unifies with `Processed[Run]`, binding
`Run = A`; the bound provider is inserted as a fact and its dependencies are
demanded recursively. This alone is complete for all non-mapped computation,
because sciline already requires a provider's argument type variables to be a
subset of its return type variables.

**Forward chaining** (the accommodation): `cyclebane.Graph.map` *relabels* the
dependents of the mapped nodes at map time (wrapping them in `MappedNode` to
mark that they carry an index; per-index duplication happens later, see Q1),
before any target is named. Dependents of a mapped generic key do not exist
yet, so `map` first runs a forward pass: rules whose argument patterns unify
with the mapped keys (or keys derived from such instantiations) are
instantiated to a fixed point. The pass is *seeded* — restricted to
instantiations that consume a mapped key or a key derived from one — because
an unrestricted pass instantiates rules for unrelated keys and creates
spurious graph sinks (this broke cyclebane's unique-sink requirement in
`reduce()` during prototyping). `output_keys()` and no-argument `visualize()`
use an unseeded forward pass to imitate the eager node set.

**Precedence**: a satisfied fact always wins over any rule (specialization
priority). Among rules, the three-tier coherence rule applies, see Q2 below.

**Declared constraints** (`TypeVar('T', int, float)`, `class Raw[Run: (A, B)]`)
are honored as *filters* during unification — a constrained type variable
refuses to bind outside its constraints — instead of driving eager
enumeration. The `constraints=` argument is removed in #237.

**Introspection taxes** (paid by any annotation-reflecting design; `main` pays
them inside `_bind_free_typevars`): bare generic classes (`def foo() -> A` for
generic `A`) are normalized to subscripted patterns; pydantic generic models
hide their type parameters from `typing.get_origin`, so unification falls back
to their generic metadata.

## Measured evidence

The existing test suite (248 tests on the #236 branch) was run against the
single-mechanism engine (initially by forcing all generics through the
demand-driven path with a two-line patch; then for real in #237):

- **242 tests pass unchanged.** All computation with explicit targets,
  map/reduce workflows, visualization with targets, setitem/copy behavior.
  Task graphs for named targets are node-for-node identical to eager
  expansion.
- Six deleted tests all tested the removed `constraints=` argument. Tests for
  constraints declared on type variables pass unchanged.
- One deliberate semantic change: a specialized provider now shadows a generic
  one regardless of insertion order. On `main`, inserting a generic provider
  *after* a specialized one replaces it for all instantiations. The new rule
  restores the specialization priority sciline had before the cyclebane
  rewrite; the common override direction (concrete over generic) is
  unaffected. Agreed as acceptable (SH, 2026-08-21).

## What it costs, honestly

Relative to `main`, the *implementation does not simplify*: the diff is
roughly +341/−87 lines in `src/`. The eager mechanism was ~60 lines of dumb,
predictable code with a strong architectural property: after `insert()`,
nothing in the library knows generics exist. The demand-driven engine replaces
that with a unification module, a template registry, two chaining directions,
and expansion hooks in five call sites (`__getitem__`, `map`, `to_task_graph`,
`output_keys`, the error-message path).

The invariant weakens from "the graph is concrete and complete" to "the graph
is concrete but complete only after expansion". Every current and future call
site that reads the graph must remember to expand; a forgotten hook does not
error — it silently sees fewer nodes.

Failure timing moves: unsatisfiable generic setups fail at graph build (as an
unsatisfied requirement) instead of at construction. Out-of-range
specializations are silently absent instead of loudly rejected — consistent
with sciline not checking parameter values against their claimed types
(see ADR 0001).

What simplifies is everything *around* the implementation:

- The user model: the concepts "constraint", "constrained TypeVar", and
  "Pipeline constraints" disappear. A pipeline is providers + params; what you
  ask for gets built. PEP 695 syntax works as-is.
- `sl.Scope` becomes deletable (fixes #233), removing the mandatory mixin base
  class for domain types.
- The built graph contains no cross-product, only demanded nodes.
- Static checking: `Processed[SampleRun]` is meaningful to mypy in exactly the
  form users write it.

The trade, stated plainly: ~250 lines of engine complexity bought back as
user-model simplicity. For a library whose users outnumber its engine readers,
that trade is favorable — but it is a trade, not a simplification.

## Open design questions

Green-field, demand-driven resolution by unification is the textbook mechanism
for this problem shape — it is how logic programming, Rust trait resolution,
and Haskell instance selection work. Eager enumeration was an artifact of
bolting generics onto a concrete-graph substrate, and constraints were its
support structure. However, the prototype is not the minimal green-field
design. Three deviations are on the table.

### Q1: Forward chaining, or deferred mapped-labeling in cyclebane?

Forward chaining plus seeding is the hairiest part of the engine — the
fixed-point loop, the seed-derivation tracking, the union-semantics
over-instantiation risk (where the only real bug so far lived), and the
special final pass for generic values. It exists for exactly one reason:
`map()` transforms the graph before any target is named.

**Corrected premise** (from reading cyclebane, 2026-08-21): cyclebane is
already a record-and-compile design. `Graph.map` does *not* duplicate
dependents; it (a) merges the array-like values and their indices into
`Graph._node_values`, kept on the side, and (b) symbolically *relabels* the
mapped roots' descendants as `MappedNode(name, indices)` — one node per
original node. The per-index duplication ("spelling out") happens exclusively
in `Graph.to_networkx()`, which sciline calls at task-graph build time — i.e.
at demand time already. `reduce()` likewise only adds one node plus an edge,
precomputing which indices the reduce consumes.

So the only thing `map()` does eagerly is the *labeling*, and the labeling is
derivable: a node carries exactly the indices of the mapped roots that reach
it (this is what `_find_successors` + `_node_with_indices` compute
incrementally), and the roots and their indices are already stored in
`_node_values`. The earlier "record and replay the whole map" idea is
over-engineered; the minimal change is:

**Proposal: derive mapped-labeling at compile time.**

- `Graph.graph` always keeps original node names; no `MappedNode` in the
  stored graph. `map()` shrinks to: validate roots, add root nodes, merge
  `node_values` (whose merge validation stays eager, so index conflicts still
  error at map time).
- `to_networkx()` first derives each node's index set by reachability from the
  mapped roots, then proceeds with the existing per-index cloning.
- `reduce()` stores a plain node whose attrs record what is reduced
  (`index`/`axis`/all); the reduce node's remaining indices are derived at
  compile (incoming indices minus reduced).
- `_from_orig_key` — the lookup-by-original-name convenience whose own
  code comment questions its worth — largely disappears; `__getitem__`,
  `__delitem__`, `__setitem__` lose their `MappedNode` special cases.
- Sciline's `get_mapped_node_names` scan for `MappedNode` is replaced by a
  small cyclebane API (e.g. `Graph.node_indices(name)`).

Consequence for sciline: the graph contains plain keys at all times, so
backward chaining works identically before and after `map()` — nodes
instantiated after mapping are indistinguishable from ones present before.
`_instantiate_forward`, `forward_bindings`, and the seeding logic are deleted;
the engine becomes backward-only (the green-field shape). Scope: roughly half
of cyclebane's `graph.py` (611 lines), guarded by its 143-test suite; sciline
is cyclebane's only user (SH, 2026-08-21), so this is in-house.

Open points:

- **`reduce(key=None)`**: today the unique sink is resolved when `reduce` is
  called; with rules, the sink set may grow as instantiation proceeds. Resolve
  at compile (unique sink of the demanded concrete graph, excluding reduce
  nodes), or require an explicit `key` when rules exist. Existing usage
  (`reduce(func=..., name=...)` without `key`) favors compile-time resolution.
- **Mapped roots must remain sources**: backward chaining must treat keys
  present in `node_values` as satisfied and never instantiate a rule for them.
- **Index-order compatibility**: sciline's mapped node identities
  (`NodeName(name, IndexValues(axes, values))`) depend on the per-node index
  *tuple order* that today emerges from the relabeling sequence. The
  compile-time derivation must reproduce this order deterministically (from
  the stored per-array axis positions); cyclebane's tests pin `to_networkx`
  output and must pass unchanged.
- **Groupby**: grouping state already lives in `node_values`; the
  `GroupbyGraph` path needs the same deferral treatment and is the least
  explored corner.
- Some errors move from `map()`/`reduce()` time to compile time (e.g.
  reachability-dependent validation); value/index conflicts stay eager.

Fallback remains option "keep seeded forward chaining" (status quo of #237,
measured compatible). The targets-at-map-time API variant is superseded by the
proposal — it bought the same ordering freedom at the cost of an idiom break,
which the compile-time derivation gets for free.

Coupling with Q3: without forward chaining, the unseeded forward pass behind
`output_keys()` / no-arg `visualize()` loses its engine, which strengthens the
case for rule-graph-first inspection.

### Q2: Coherence instead of last-wins among rules — decided, implemented

The prototype initially resolved multiple matching rules by "latest registered
wins", inherited from concrete-key replacement semantics. This makes
resolution order-dependent global state — the classic source of
unreproducible workflow-composition bugs. Trait/instance systems solve this
with coherence rules. The agreed three-tier rule (implemented in #237):

1. **Equal patterns** (identical up to renaming of type variables): the later
   registration *replaces* the earlier one. This is the explicit-override
   workflow (`wf.insert(my_better_loader)`) and mirrors concrete-provider
   replacement. Kept.
2. **Comparable patterns** (one strictly more specific, e.g.
   `Processed[list[Run]]` vs `Processed[Run]`): the more specific rule wins,
   regardless of order. "Strictly more specific" = its pattern is an instance
   of the other's under some substitution, but not conversely; implementable
   with one-sided unification in ~30 lines.
3. **Incomparable overlapping patterns** both matching a demanded key: error
   at demand time, naming both providers.

This preserves the two override idioms users actually rely on (replace same
pattern; specialize) and turns the silent order-dependence into a loud
`AmbiguousProvider` error. Tier 1 replacement applies across kinds: a generic
value with the same pattern replaces a generic provider and vice versa,
restoring last-write-wins where patterns are genuinely interchangeable.
Subsumption also orders constrained against unconstrained type variables: a
constrained pattern is strictly more specific than its unconstrained
counterpart. One existing test broke, and it was one that *documented* the
order-dependence (`test_multiple_matching_partial_providers_uses_latest`,
overlapping `A[int, T1]` vs `A[T2, float]`); it now asserts the error.

### Q3: Rule-graph-first inspection

The prototype makes `output_keys()`, no-argument `visualize()` and
`_repr_html_` imitate `main`'s concrete views via unseeded forward expansion.
This is where the remaining inspection-surface awkwardness lives: a pipeline
whose rules are only reachable from targets (e.g. a no-argument generic
source) shows nothing until demanded.

Green-field, the pipeline's primary representation *is* the rule graph: one
node per pattern (`Processed[Run]`), edges by matching a rule's output pattern
against other rules' input patterns. Cross-rule type-variable identity does
not exist, but display does not need it — origin matching suffices, and the
displayed variable name is just each rule's own (usually consistently spelled)
name. Concrete facts (params, concrete providers) attach to the rule graph
where their keys are instances of patterns. The concrete instantiated view
remains available per target via `get(target).visualize()`, unchanged.

This is likely the easiest of the three (rendering code plus changed
defaults). The one genuinely breaking decision is `output_keys()`: does it
return concrete keys (forward-expanded, status quo), patterns for rules plus
concrete sinks (breaking for `compute(pl.output_keys())`), or does the rule
view get a separate accessor while `output_keys()` keeps its meaning? Open;
affects `visualize()`'s no-argument default via `tp=self.output_keys()`.

## Decision log

| Date | Decision | Where |
|------|----------|-------|
| 2026-08-21 | Demand-driven instantiation is viable; PEP 695 needs no constraints mechanism. Prototyped. | #236 |
| 2026-08-21 | Generic params (`params={Raw: value}`) supported as rules matched on demand. | #236 (review), commit 2fc0461 |
| 2026-08-21 | Specialized-provider-shadows-generic accepted as a semantic change; generic-replaces-specialized not worth preserving. | #237, this doc |
| 2026-08-21 | Q2 decided: three-tier coherence (replace equal patterns, most-specific wins, incomparable overlap errors) instead of last-wins. Implemented. | #237 |
| open | Single mechanism (#237) vs. coexistence (#236) vs. separate class. | discussion |
| open | Q1 (forward chaining vs. deferred mapped-labeling in cyclebane), Q3 (rule-graph inspection). | this doc |

## References

- Issues: [#233](https://github.com/scipp/sciline/issues/233) (Scope metaclass
  conflict), [#235](https://github.com/scipp/sciline/issues/235) (PEP 695
  support).
- Prototypes: [#236](https://github.com/scipp/sciline/pull/236) (demand-driven
  alongside eager), [#237](https://github.com/scipp/sciline/pull/237)
  (demand-driven only; net −118 lines vs #236, +254 vs `main`).
- Coexistence-cost analysis:
  [#236 comment](https://github.com/scipp/sciline/pull/236#issuecomment-5365975740).
- Compatibility simulation: force all generics through the demand-driven path
  by replacing the eager-dispatch condition in `DataGraph.insert` and
  `__setitem__` with `False`, then run the full test suite.
- Pre-cyclebane sciline resolved generics on demand (concrete-over-generic
  priority); the eager design arrived with the cyclebane rewrite
  (see `rewrite.ipynb` in this directory).
