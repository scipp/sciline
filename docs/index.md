# Sciline

<span style="font-size:1.2em;font-style:italic;color:#5a5a5a">
  Build scientific pipelines for your data
  </br></br>
</span>

## Why Sciline?

Writing, testing, and maintaining complex data analysis workflows is hard.
Boiler-plate code may hide the actual analysis, making it hard to understand.
The code may be hard to test, leading to bugs.

Systems like [Snakemake](https://snakemake.readthedocs.io/en/stable/) can help with this, by using a set of *rules* declaring their inputs and outputs.
A rule might run, e.g., a Python script.
Snakemake then automatically assembles a task graph, and runs the tasks in the correct order to compute the desired outputs.
But how do you write the Python script?
It in itself can be thought of as a workflow.
It may have a significant number of inputs and outputs, and may be complex with many internal computation steps.
If intermediate results are large, splitting the rule into multiple rules may be prohibitive due to the overhead of repeatedly writing to and reading from disk.

Regardless of whether you are using Snakemake, another workflow management system, or are just writing Python scripts or Jupyter notebooks, Sciline can help you.

Sciline takes a *declarative* approach, inspired by [dependency injection](https://en.wikipedia.org/wiki/Dependency_injection) frameworks.
By relying on Python's type-hinting, a [domain-specific language](https://en.wikipedia.org/wiki/Domain-specific_language) is used to describe the workflow.
This enforces a clear expression of intent, aiding readability and enabling automatic assembly of a task graph that can compute desired outputs.

## FAQ

### Is Sciline an alternative to Snakemake?

Sciline is not intended to replace Snakemake or similar systems.
We see Snakemake as a complementary tool that operates on a higher level.
While Snakemake's rules describe transformations of *files*, Sciline used Python functions to describe transformations of *Python objects* such as NumPy arrays or Pandas dataframes.

### But I do not want to change all my code!

Sciline is very non-invasive.
If you are willing to define a few types (or type aliases) and add [type annotations](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) to your functions &mdash; which is considered good practice anyway &mdash; you can use it with Sciline with just a few extra lines of code.
Your existing Python code can remain generic and can still be used without Sciline.

## At a glance

### Install

Pip:

```sh
pip install sciline
```

Conda:

```sh
conda install -c conda-forge -c scipp sciline
```

### Use

```python
from typing import NewType
import sciline

# 1. Define domain types

Filename = NewType('Filename', str)
RawData = NewType('RawData', dict)
CleanedData = NewType('CleanedData', list)
ScaleFactor = NewType('ScaleFactor', float)
Result = NewType('Result', float)


# 2. Define providers


def load(filename: Filename) -> RawData:
    """Load the data from the filename."""
    return {'data': [1, 2, float('nan'), 3], 'meta': {'filename': filename}}


def clean(raw_data: RawData) -> CleanedData:
    """Clean the data, removing NaNs."""
    import math

    return [x for x in raw_data['data'] if not math.isnan(x)]


def process(data: CleanedData, param: ScaleFactor) -> Result:
    """Process the data, multiplying the sum by the scale factor."""
    return sum(data) * param


# 3. Create pipeline

providers = [load, clean, process]
params = {Filename: 'data.txt', ScaleFactor: 2.0}
pipeline = sciline.Pipeline(providers, params=params)

pipeline.compute(Result)
```

See [Getting Started](user-guide/getting-started) for context and explanation.


```{toctree}
---
hidden:
---

user-guide/index
recipes/index
api-reference/index
developer/index
about/index
```
