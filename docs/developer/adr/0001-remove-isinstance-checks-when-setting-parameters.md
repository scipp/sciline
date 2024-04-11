# ADR 0001: Remove isinstance checks when setting parameters

- Status: proposed
- Deciders:
- Date: 2024-04-11

## Context

Sciline builds a data dependency graph based on type hints of callables.
Dependencies can be fulfilled by setting values (instances of classes) as so called *parameters*.
In an attempt to extend the correctness guarantees of the dependency graph, Sciline's `__setitem__` checks if the value is instance of the key (a type) when setting a parameter.

This has led to a number of problems.
For example, supporting different file handles types is too difficult [#140](https://github.com/scipp/sciline/issues/140),
parameter type handling is too inflexible in general [#144](https://github.com/scipp/sciline/issues/144),
and the mechanism is broken with Python 3.12 type aliases [#145](https://github.com/scipp/sciline/issues/145).
In short, the mechanism gets in the way of the user, since it causes false positives.

Considering the bigger picture, we can think of this mechanism as a poor man's form of *validation*.
Validation of input parameters is very important when running workflows, but it should be done in a more explicit way.
Validating the type is only a fraction of what we want to do when validating parameters.
Therefore, we should remove this mechanism and replace it with a more general validation mechanism.
The more general validation mechanism can be considered out of scope for Sciline, and should be implemented in the user code or using other common libraries such as `pydantic`.

Finally, we can think of this mechanism as a form of runtime type checking.
This is not the intended scope of Sciline.

## Decision

- Remove the mechanism that checks if a value is an instance of the key when setting it as a parameter.
- Encourage users to validate inputs in providers, which can also be tested in unit tests without setting up the full workflow.
- Encourage users to use a more general parameter validation mechanism using other libraries.

## Consequences

### Positive

- The mechanism will no longer get in the way of the user.
- The code will be simplified slightly.

### Negative

- The correctness guarantees will be slightly diminished.