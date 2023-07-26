# Sciline

<span style="font-size:1.2em;font-style:italic;color:#5a5a5a">
  Build scientific pipelines for your data
  </br></br>
</span>

## Why Sciline?

Writing, testing, and maintaining complex data analysis workflows is hard.
Boiler-plate code may hide the actual analysis, making it hard to understand.
The code may be hard to test, leading to bugs.

Sciline takes a *declarative* approach, inspired by [dependency injection](https://en.wikipedia.org/wiki/Dependency_injection) frameworks.
By relying on Python's type-hinting, a [domain-specific language](https://en.wikipedia.org/wiki/Domain-specific_language) is used to describe the workflow.
This enforces a clear expression of intent, aiding readability and enabling automatic assembly of a task graph that can compute desired outputs.

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
api-reference/index
developer/index
```
