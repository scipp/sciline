# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

import sciline as sl


def test_raises_with_zero_columns() -> None:
    with pytest.raises(ValueError):
        sl.ParamTable(row_dim=int, columns={})


def test_raises_with_inconsistent_column_sizes() -> None:
    with pytest.raises(ValueError):
        sl.ParamTable(row_dim=int, columns={int: [1, 2, 3], float: [1.0, 2.0]})


def test_raises_with_inconsistent_index_length() -> None:
    with pytest.raises(ValueError):
        sl.ParamTable(row_dim=int, columns={float: [1.0, 2.0]}, index=[1, 2, 3])


def test_raises_with_non_unique_index() -> None:
    with pytest.raises(ValueError):
        sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}, index=[1, 1, 2])


def test_contains_includes_all_columns() -> None:
    pt = sl.ParamTable(row_dim=int, columns={int: [1, 2, 3], float: [1.0, 2.0, 3.0]})
    assert int in pt
    assert float in pt
    assert str not in pt


def test_contains_does_not_include_index() -> None:
    pt = sl.ParamTable(row_dim=int, columns={float: [1.0, 2.0, 3.0]})
    assert int not in pt


def test_len_is_number_of_columns() -> None:
    pt = sl.ParamTable(row_dim=int, columns={int: [1, 2, 3], float: [1.0, 2.0, 3.0]})
    assert len(pt) == 2


def test_defaults_to_range_index() -> None:
    pt = sl.ParamTable(row_dim=int, columns={float: [1.0, 2.0, 3.0]})
    assert pt.index == [0, 1, 2]


def test_index_with_no_columns() -> None:
    pt = sl.ParamTable(row_dim=int, columns={}, index=[1, 2, 3])
    assert pt.index == [1, 2, 3]
    assert len(pt) == 0
