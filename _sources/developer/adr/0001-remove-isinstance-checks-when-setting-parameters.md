# ADR 0001: Remove isinstance checks when setting parameters

- Status: accepted
- Deciders: Jan-Lukas, Neil, Simon
- Date: 2024-04-15

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
We should ask ourselves if this is the intended scope of Sciline.
If it is, shouldn't we also check that each provider actually returns the correct type?

The main problem with not checking value types when setting parameters is that it is not possible to catch such errors with `mypy`, in contrast to return values of providers, which `mypy` *can* check.

Consider the following example of setting $Q$ bins for a workflow, given by a `scipp.Variable`, which would then be passed to `scipp.hist` to create a histogram:

```python
pipeline[QBins] = sc.linspace(...)
pipeline[QBins] = 1000  # error in current implementation
pipeline[QBins] = sc.linspace(..., unit='m')  # no error, but wrong unit
```

Checking the type catches the first error, but not the second.
Paradoxically, setting an integer would often be a valid operation in the example, since `scipp.hist` can handle this case, whereas the wrong unit would not be valid.
This may indicate that defining `QBins` as an alias of `scipp.Variable` is actually an instance of an anti-pattern.
Instead, imagine we have defined a specific `class QBins`, which performs validation in its constructor, and defines `__call__` so it can be used as a provider:

```python
pipeline.insert(QBins(sc.linspace(...)))
pipeline.insert(QBins(1000))  # ok
pipeline.insert(QBins(sc.linspace(..., unit='m')))  # error constructing QBins
```

This example illustrates that a clearer and more specific expression of intent can avoid the need for relying on checking the type of the value when setting a parameter.

## Decision

- The core scope of Sciline is the definition of task graphs.
  Type validation is not.
- Remove the mechanism that checks if a value is an instance of the key when setting it as a parameter.
- Encourage users to validate inputs in providers, which can also be tested in unit tests without setting up the full workflow.
- Encourage users to use a more general parameter validation mechanism using other libraries.
- Consider adding a mechanism to inject a callable to use for parameter validation as a argument when creating a `Pipeline`.

## Consequences

### Positive

- The mechanism will no longer get in the way of the user.
- The code will be simplified slightly.

### Negative

- `sciline.Pipeline` will support duck-typing for parameters, in a way that cannot be checked with `mypy`.
