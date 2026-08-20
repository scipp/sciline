# Workflow Design

## Context

This document discusses problems and requirements for data-reduction of neutron-scattering experiments which have led to the design of Sciline.
The terminology used in the examples is specific to neutron-scattering experiments, but the concepts are general and can be applied to other domains.

## Introduction

### Traditional data-reduction workflows

Traditionally, we have supplied users with a toolbox of algorithms and optionally a reduction script or a notebook that uses those algorithms.
Conceptually this looks similar to the following:

```python
# Define parameters
sample_run_id = 12345
background_run_id = 12300
direct_beam_filename = 'direct_beam.h5'
wav_bins = linspace(...)
Q_bins = linspace(...)

# Load data
sample = load_sample(run=sample_run_id)
background = load_background(run=background_run_id)
direct_beam = load_direct_beam(run=direct_beam_filename)

# Process
mask_detectors(sample)
mask_detectors(background)
sample_monitors = preprocess_monitors(sample, wav_bins)
background_monitors = preprocess_monitors(background, wav_bins)

transmission_fraction = transmission_fraction(**sample_monitors)
sample_iofq = compute_i_of_q(
    sample,
    direct_beam,
    transmission_fraction,
    Q_bins)

transmission_fraction = transmission_fraction(**background_monitors)
background_iofq = compute_i_of_q(
    background,
    direct_beam,
    transmission_fraction,
    Q_bins)

iofq = subtract_background(sample_iofq, background_iofq)
```

This is an *imperative workflow*, where the user specifies the order of operations and the dependencies between them.
This is not ideal for a number of reasons:

- The user has to know the order of operations and the dependencies between them.
- The user has to know which algorithms to use.
- The user has to know which parameters to use for each algorithm.
- The user has to know which data to use for each algorithm.
- The user can easily introduce mistakes into a workflow, e.g., by using the wrong order of operations, or by overwriting data.
  This is especially problematic in Jupyter notebooks, where the user can easily run cells out of order.

Our most basic programming models provide little help to the user.
For example we typically write components of reduction workflows as functions of `scipp.Variable` or `scipp.DataArray` objects:

```python
def transmission_fraction(
    incident_monitor: sc.DataArray,
    transmission_monitor: sc.DataArray,
) -> sc.DataArray:
"""
Compute transmission fraction from incident and transmission monitors.

Parameters
----------
incident_monitor:
    Incident monitor.
transmission_monitor:
    Transmission monitor.
"""
    return transmission_monitor / incident_monitor
```

Here, we rely on naming of function parameters as well as docstrings to convey the meaning of the parameters, and it is up to the user to pass the correct inputs.
While good practices such as keyword-only arguments can help, this is still far from a scalable and maintainable solution.

As an improvement, we could adopt an approach with more specific domain types, e.g.,

```python
def transmission_fraction(
    incident_monitor: IncidentMonitor,
    transmission_monitor: TransmissionMonitor,
) -> TransmissionFraction:
    return TransmissionFraction(transmission_monitor / incident_monitor)
```

We could now run [mypy](https://mypy-lang.org/) on reduction scripts to ensure that the correct types are passed to each function.
However, this is not practical with dynamic workflows, i.e., when users modifying workflows in a Jupyter notebooks on the fly.
Aside from this, such an approach would still not help with several of the other issues listed above.


### High-level summary of proposed approach

We propose an architecture combining *domain-driven design* with *dependency injection*.
Dependency injection aids in building a declarative workflow.
We define domain-specific concepts that are meaningful to the (instrument) scientist.
Simple functions provide workflow components that define relations between these domain concepts.

Concretely, we propose to define specific domain types, such as `IncidentMonitor`, `TransmissionMonitor`, and `TransmissionFraction` in the example above.
However, instead of the user having to pass these to functions, we use dependency injection to provide them to the functions.
In essence this will build a workflow's task graph.


### Domain-driven design

Domain-Driven Design (DDD) is an approach to software development that aims to make software more closely match the domain it is used in.
The obvious benefit of this is that it makes it easier for domain experts to understand and modify the software.

How should we define the domain for the purpose of data reduction?
Looking at, e.g., [Mantid](https://www.mantidproject.org/), we see that the domain is defined as "data reduction for any type of neutron scattering experiment".
This has led to more than 1000 algorithms, making it hard for users to know how to use them.
Furthermore, while algorithms provide some sort of domain-specific language, the data types are generic.

What we propose here is to define the domain more narrowly, highly specific to a technique or even specific to an instrument or experiment.
This will reduce the scope to cover in the domain-specific language.
By making data types specific to the domain, we provide nouns for the domain-specific language.


### Dependency injection

Dependency injection is a common technique for implementing the [inversion of control](https://en.wikipedia.org/wiki/Inversion_of_control) principle.
It makes components of a system more loosely coupled, and makes it easier to replace components, including for testing purposes.
Dependency injection can be performed manually, but there are also frameworks that can help with this.

From the [Guice documentation](https://github.com/google/guice/wiki/MentalModel#injection) (Guice is a dependency injection framework for Java):

> "This is the essence of dependency injection. If you need something, you don't go out and get it from somewhere, or even ask a class to return you something. Instead, you simply declare that you can't do your work without it, and rely on Guice to give you what you need.
>
> This model is backwards from how most people think about code: it's a more *declarative model* rather than an *imperative one*. This is why dependency injection is often described as a kind of inversion of control (IoC)."
> (emphasis added)

## Architecture

### In a nutshell

1. The user will define building blocks of a workflow using highly specific domain types for the type-hints, such as `IncidentMonitor`, `TransmissionMonitor`, and `TransmissionFraction`, e.g.,

   ```python
   def transmission_fraction(
       incident_monitor: IncidentMonitor,
       transmission_monitor: TransmissionMonitor,
   ) -> TransmissionFraction:
       return TransmissionFraction(transmission_monitor / incident_monitor)
   ```

2. The user passes a set of building blocks to the system, which assembles a dependency graph based on the type-hints.
3. The user requests a specific output from the system using one of the domain types.
   This may be computed directly, or the system may construct a `dask` graph to compute the output.

Depending on the level of expertise of the user and the level of control they need, step 1.) or step 1.) and 2.) will be omitted, as pre-defined building blocks and sets of building blocks can be provided in domain-specific support-libraries for common use cases.

### Parameter handling

Generally, the user must provide configuration parameters to a workflow.
In many cases there are defaults that can be used.
In either case, these parameters must be associated with the correct step in the workflow.
This gets complicated by the non-linear nature of the workflow.
A flat list of parameters has been used traditionally, relying entirely on parameter naming.
This is problematic for two reasons:
First, certain basic workflow steps may be used in multiple places.
Second, workflows frequently contain nested steps, which may have the same parameters (or not).
This makes the process of setting parameters somewhat opaque and error-prone.
Furthermore, it relies on a hand-written higher-level workflow to set parameters for nested steps, mapping between globally-uniquely-named parameters and the parameters of the nested steps.
These, in turn, require complicated testing.

A hierarchical parameter system could provide an alternative, but makes it harder to set "global" parameters.
For example, we may want to use the same wavelength-binning for all steps in the workflow.

We propose to handle parameters as dependencies of workflow steps.
That is, the dependency-injection system is used to provide parameters to workflow steps.
Parameters are identified via their type, i.e., we will require defining a domain-specific type for each parameter, such as `WavelengthBinning`.

For nested workflows, we can use a child injector, which provides a scope for parameters.
Parent-scopes can be searched for parameters that are not found in the child-scope, providing a mechanism for "global" parameters.

*Note:
The idea of using child injectors was discarded during implementation for a number of reasons.
Parameters in a "nested" scope can now be realized using the mechanism of generic providers.*

### Metadata handling

There have been a number of discussions around metadata handling.
For example, the support (or non-support) of an arbitrary `attrs` dict as part of `scipp.Variable` and `scipp.DataArray`.
Furthermore, we may have metadata that is part of the data-catalog, which may partially overlap with the metadata that is part of the data itself.
The current conclusion is that any attempt to handle metadata in a generic and automatic way will not work.
Therefore, if a user wants to provide metadata for a workflow result, they must do so *explicitly* by specifying functions that can assemble that metadata.
As with regular results, this can be done by injecting the input metadata into the function that computes the result's metadata.

### Domain-specific types

The system will use domain-specific types to identify workflow steps and parameters.
The system does not require subclassing or use of decorators.
Domain-specific types can be defined as regular classes or subclasses of existing types.
In many cases, a simple alias will be sufficient.

`typing.NewType` can be used as a simple way of creating a domain-specific type alias for type-checking.
For example, we can use it to create a type for a `scipp.DataArray` that represents a transmission monitor.
This avoids a more complex solution such as creating a wrapper class or a subclass:

```python
import typing

TransmissionMonitor = typing.NewType('TransmissionMonitor', scipp.DataArray)
```

Note that this does not create a new type, but rather a new name for an existing type.
That is, `isinstance(monitor, TransmissionMonitor)` does not work, since `TransmissionMonitor` is not a type.
Furthermore, operations will revert to the underlying type, e.g., `monitor * 2` will return a `scipp.DataArray`.
For this application this would actually be desired behavior:
Applying an operation to a domain type will generally result in a different type, so falling back to the underlying type is the correct behavior and forces the user to be explicit about the type of the result.

## Notes

- Earlier versions of this design document included a detailed discussion on how to handle nested workflows using child-injectors.
 During initial implementation efforts this idea was discarded, as it was found to be too complicated and not necessary.
- Earlier versions of this design document included considerations on the use of the [injector](https://injector.readthedocs.io/en/latest/) Python library.
  During early implementation efforts it was found that this library did not provide advantages beyond the very basic features (which can be implemented in a few lines of code).
  For example, it was attempted to use `injector`'s `multibind` and child-injector mechanism to handle nested workflows as well as multiple runs.
  This turned out to be way too complicated.
  By using a dedicated home-grown solution, in particular using generic providers and parameter tables, we found a more straightforward solution and furthermore avoided introducing a dependency.