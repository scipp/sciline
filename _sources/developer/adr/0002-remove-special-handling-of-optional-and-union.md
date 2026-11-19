# ADR 0002: Remove special handling of Optional and Union

- Status: accepted
- Deciders: Jan-Lukas, Johannes, Mridul, Simon, Sunyoung
- Date: 2024-04-15

## Context

### General

Sciline builds a data dependency graph based on type hints of callables.
Some callables may have optional inputs, which are commonly represented by `Optional[T]` in the type hint, for some type `T`.
Therefore, in [#50](https://github.com/scipp/sciline/pull/50) we have added special handling for `Optional` and [#89](https://github.com/scipp/sciline/pull/89) extended this for `Union`.
In the case of `Optional`, they way this works is that `sciline.Pipeline` prunes branches at the node where the optional input used, if any ancestor node has unsatisfied dependencies.
Instead, an implicit `None` provider is added.
This has a series of problems, which we exemplify for the case of `Optional`.

1. Default values (which are currently ignored by Sciline) are overridden by the implicit `None` provider.
   In other words, Sciline assumes that the default value of the optional input is `None`.
2. Entire branches are pruned, which can hide bugs.
   If the users added providers for the optional input, they will not be used if any of them has unintentionally unsatisfied dependencies.
3. The special mechanism prevents the (in principle very valid) use of any providers that return an `Optional` or `Union` type.
4. Optional inputs cannot be set to `None` *explicitly*.

In summary, the special handling of `Optional` and `Union` is too implicit and causes more problems than it solves.
There are a couple more aspects to consider.

### Readability of user code

Handling `Optional` explicitly would make user code more readable.
Consider the following example:

```python
pipeline[MyParam] = 1.2
```

In the current implementation this gives no indication to the user that `MyParam` is not a required input.
Furthermore, if the line is removed, the user may not realize that `MyParam` is available as an optional input.
With the proposed change, the user can make this explicit:

```python
pipeline[Optional[MyParam]] = 1.2
```

Above it is clear that `MyParam` is optional, and it can be set to `None` explicitly:

```python
pipeline[Optional[MyParam]] = None
```

### Code complexity and maintainability

The special handling of `Optional` and `Union` is a significant source of complexity in the code, requiring a significant amount of unit testing.

### Conceptual clarity

The current redesign of Sciline highlighted that the current implementation is conceptually flawed.
It makes it tricky to represent the internals of `sciline.Pipeline` as a simple data dependency graph.
The special handling of `Optional` and `Union` seems to require pervasive changes to the code, which is a sign that it is not a good fit for the design.

### Counter arguments

#### Multiple providers may depend on the same input, but not all optionally

This seems like a special case that we have not seen in practice, is likely not worth the complexity of the current implementation.

#### Using a provider returning a non-optional output to fulfill an optional input

This is a very valid use case, but it would be made impossible if we stop associating a node `T` with an optional input `Optional[T]`.
There are a couple of possible workarounds:

- Add an explicit `Optional` provider that wraps (or depends on) the non-optional provider.
- Modify the graph structure (which we plan to support in the redesign of Sciline) using something like `pipeline[Optional[MyParam]] = pipeline[MyParam]`.

#### Using a provider to return one of a union's types

Same as above, for `Optional[T]`.

#### Setting union parameters is unwieldy

Given a provider `f(x: A | B | C) -> D: ...`, a user would need to set a value for the input of `f` like `pipeline[A | B | C] = ...`.
It would be easier if they could be more specific, like `pipeline[A] = ...`.

In this case, we think defining an alias for `A | B | C` would be a better solution than the current special handling of `Union`.
It would force the user to be more explicit about the input type, which is a good thing.
Conceptually the use of `Union` may just be an indicator that `f` depends on some common aspect of `A`, `B`, and `C`, which could be made explicit by defining a new type or protocol.

## Decision

Remove the special handling of `Optional` and `Union`.

## Consequences

### Positive

- Sciline's code will be simplified significantly.
- User code will be more readable.
- Implicit behavior around pruning and using `None` providers will be removed.
- Users can use providers that return `Optional` or `Union` types.
- Decouples the handling of optional inputs from the handling of default values.
  This will enable us to make independent decisions about how to handle default values.

### Negative

- Workarounds are needed for the use case of using a provider returning a non-optional output to fulfill an optional input, and for setting union parameters.