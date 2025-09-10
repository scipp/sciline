from typing import NewType

import numpy as np
import pandas as pd

import sciline as sl

_fake_filesytem = {
    'file102.txt': [1, 2, float('nan'), 3],
    'file103.txt': [4, 5, 6, 7],
    'file104.txt': [8, 9, 10, 11, 12],
    'file105.txt': [13, 14, 15],
}

# 1. Define domain types

Filename = NewType('Filename', str)
RawData = NewType('RawData', dict)
CleanedData = NewType('CleanedData', list)
ScaleFactor = NewType('ScaleFactor', float)
Result = NewType('Result', float)
Material = NewType('Material', str)


# 2. Define providers


def load(filename: Filename) -> RawData:
    """Load the data from the filename."""

    data = _fake_filesytem[filename]
    return RawData({'data': data, 'meta': {'filename': filename}})


def clean(raw_data: RawData) -> CleanedData:
    """Clean the data, removing NaNs."""
    import math

    return CleanedData([x for x in raw_data['data'] if not math.isnan(x)])


def process(data: CleanedData, param: ScaleFactor) -> Result:
    """Process the data, multiplying the sum by the scale factor."""
    return Result(sum(data) * param)


def merge(*data):
    return sum(data)


def test_groupby_material_at_result():
    # Create pipeline
    providers = [load, clean, process]
    params = {ScaleFactor: 2.0}
    base = sl.Pipeline(providers, params=params)

    # Make parameter table
    run_ids = [102, 103, 104, 105]
    sample = ['diamond', 'graphite', 'graphite', 'graphite']
    filenames = [f'file{i}.txt' for i in run_ids]
    param_table = pd.DataFrame(
        {Filename: filenames, Material: sample}, index=run_ids
    ).rename_axis(index='run_id')

    # Group by Material and merge Result
    grouped = (
        base.map(param_table)
        .groupby(Material)
        .reduce(key=Result, func=merge, name="merged")
    )

    result = sl.compute_mapped(grouped, "merged")
    assert result['diamond'] == 12.0
    assert result['graphite'] == 228.0


def test_groupby_material_at_rawdata():
    # Create pipeline
    providers = [load, clean, process]
    params = {ScaleFactor: 2.0}
    base = sl.Pipeline(providers, params=params)

    # Make parameter table
    run_ids = [102, 103, 104, 105]
    sample = ['diamond', 'graphite', 'graphite', 'graphite']
    filenames = [f'file{i}.txt' for i in run_ids]
    param_table = pd.DataFrame(
        {Filename: filenames, 'Material': sample}, index=run_ids
    ).rename_axis(index='run_id')

    # Define function to merge RawData
    def merge_raw(*das):
        out = {"data": [], "meta": {}}
        for da in das:
            out["data"].extend(da["data"])
            for k, v in da["meta"].items():
                if k not in out["meta"]:
                    out["meta"][k] = []
                out["meta"][k].append(v)
        return out

    # Group by Material and merge RawData
    grouped = (
        base[RawData]
        .map(param_table)
        .groupby('Material')
        .reduce(key=RawData, func=merge_raw, name='merged')
    )

    # Join back to base pipeline
    new = base.copy()
    new[RawData] = None

    mapped = new.map(
        # Need dummy RawData column to allow re-attaching
        pd.DataFrame({RawData: [1, 2], 'Material': ['diamond', 'graphite']}).set_index(
            'Material'
        )
    )

    # Attach the grouped merged data to the lower part of the pipeline
    mapped[RawData] = grouped['merged']

    clean_data = sl.compute_mapped(mapped, CleanedData)
    assert np.array_equal(clean_data['diamond'], [1, 2, 3])
    assert np.array_equal(
        clean_data['graphite'], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )

    result = sl.compute_mapped(mapped, Result)
    assert result['diamond'] == 12.0
    assert result['graphite'] == 228.0

    raw_data = sl.compute_mapped(mapped, RawData, index_names=['Material'])
    assert np.array_equal(
        raw_data['diamond']['data'], [1, 2, float('nan'), 3], equal_nan=True
    )
    assert np.array_equal(
        raw_data['graphite']['data'],
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        equal_nan=True,
    )
    assert raw_data['diamond']['meta'] == {'filename': ['file102.txt']}
    assert raw_data['graphite']['meta'] == {
        'filename': ['file103.txt', 'file104.txt', 'file105.txt']
    }
