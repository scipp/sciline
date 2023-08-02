# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import pytest

import sciline as sl
from sciline.pipeline import Item, Label


def test_set_param_table_raises_if_param_names_are_duplicate():
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError):
        pl.set_param_table(sl.ParamTable(str, {float: [1.0, 2.0, 3.0]}))
    assert str not in pl.param_tables


def test_set_param_table_raises_if_row_dim_is_duplicate():
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError):
        pl.set_param_table(sl.ParamTable(int, {str: ['a', 'b', 'c']}))
    assert pl.param_tables[int] == sl.ParamTable(int, {float: [1.0, 2.0, 3.0]})


def test_can_get_elements_of_param_table() -> None:
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(Item((Label(int, 1),), float)) == 2.0


def test_can_depend_on_elements_of_param_table() -> None:
    def use_elem(x: Item((Label(int, 1),), float)) -> str:
        return str(x)

    pl = sl.Pipeline([use_elem])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(str) == "2.0"


def test_can_compute_series_of_param_values() -> None:
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(sl.Series[int, float]) == {0: 1.0, 1: 2.0, 2: 3.0}


def test_can_compute_series_of_derived_values() -> None:
    def process(x: float) -> str:
        return str(x)

    pl = sl.Pipeline([process])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(sl.Series[int, str]) == {0: "1.0", 1: "2.0", 2: "3.0"}


def test_explicit_index_of_param_table_is_forwarded_correctly() -> None:
    def process(x: float) -> int:
        return int(x)

    pl = sl.Pipeline([process])
    pl.set_param_table(
        sl.ParamTable(str, {float: [1.0, 2.0, 3.0]}, index=['a', 'b', 'c'])
    )
    assert pl.compute(sl.Series[str, int]) == {'a': 1, 'b': 2, 'c': 3}


def test_can_gather_index() -> None:
    Sum = NewType("Sum", float)
    Name = NewType("Name", str)

    def gather(x: sl.Series[Name, float]) -> Sum:
        return Sum(sum(x.values()))

    def make_float(x: str) -> float:
        return float(x)

    pl = sl.Pipeline([gather, make_float])
    pl.set_param_table(sl.ParamTable(Name, {str: ["1.0", "2.0", "3.0"]}))
    assert pl.compute(Sum) == 6.0


def test_can_zip() -> None:
    Sum = NewType("Sum", float)
    Str = NewType("Str", str)
    Run = NewType("Run", int)

    def gather_zip(x: sl.Series[Run, Str], y: sl.Series[Run, int]) -> Sum:
        z = [f'{x_}{y_}' for x_, y_ in zip(x.values(), y.values())]
        return Sum(str(z))

    def use_str(x: str) -> Str:
        return Str(x)

    pl = sl.Pipeline([gather_zip, use_str])
    pl.set_param_table(sl.ParamTable(Run, {str: ['a', 'a', 'ccc'], int: [1, 2, 3]}))

    assert pl.compute(Sum) == "['a1', 'a2', 'ccc3']"


def test_diamond_dependency_pulls_values_from_columns_in_same_param_table() -> None:
    Sum = NewType("Sum", float)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Row = NewType("Run", int)

    def gather(
        x: sl.Series[Row, float],
    ) -> Sum:
        return Sum(sum(x.values()))

    def join(x: Param1, y: Param2) -> float:
        return x / y

    pl = sl.Pipeline([gather, join])
    pl.set_param_table(sl.ParamTable(Row, {Param1: [1, 4, 9], Param2: [1, 2, 3]}))

    assert pl.compute(Sum) == 6


def test_dependencies_on_different_param_tables_broadcast() -> None:
    Row1 = NewType("Row1", int)
    Row2 = NewType("Row2", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Product = NewType("Product", str)

    def gather_both(x: sl.Series[Row1, Param1], y: sl.Series[Row2, Param2]) -> Product:
        broadcast = [[x_, y_] for x_ in x.values() for y_ in y.values()]
        return str(broadcast)

    pl = sl.Pipeline([gather_both])
    pl.set_param_table(sl.ParamTable(Row1, {Param1: [1, 2, 3]}))
    pl.set_param_table(sl.ParamTable(Row2, {Param2: [4, 5]}))
    assert pl.compute(Product) == "[[1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5]]"


def test_dependency_on_other_param_table_in_parent_broadcasts_branch() -> None:
    Row1 = NewType("Row1", int)
    Row2 = NewType("Row2", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Summed2 = NewType("Summed2", int)
    Product = NewType("Product", str)

    def gather2_and_combine(x: Param1, y: sl.Series[Row2, Param2]) -> Summed2:
        return Summed2(x * sum(y.values()))

    def gather1(x: sl.Series[Row1, Summed2]) -> Product:
        return str(list(x.values()))

    pl = sl.Pipeline([gather1, gather2_and_combine])
    pl.set_param_table(sl.ParamTable(Row1, {Param1: [1, 2, 3]}))
    pl.set_param_table(sl.ParamTable(Row2, {Param2: [4, 5]}))
    assert pl.compute(Product) == "[9, 18, 27]"


def test_dependency_on_other_param_table_in_grandparent_broadcasts_branch() -> None:
    Row1 = NewType("Row1", int)
    Row2 = NewType("Row2", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Summed2 = NewType("Summed2", int)
    Combined = NewType("Combined", int)
    Product = NewType("Product", str)

    def gather2(x: sl.Series[Row2, Param2]) -> Summed2:
        return Summed2(sum(x.values()))

    def combine(x: Param1, y: Summed2) -> Combined:
        return Combined(x * y)

    def gather1(x: sl.Series[Row1, Combined]) -> Product:
        return str(list(x.values()))

    pl = sl.Pipeline([gather1, gather2, combine])
    pl.set_param_table(sl.ParamTable(Row1, {Param1: [1, 2, 3]}))
    pl.set_param_table(sl.ParamTable(Row2, {Param2: [4, 5]}))
    assert pl.compute(Product) == "[9, 18, 27]"


def test_generic_providers_work_with_param_tables() -> None:
    Param = TypeVar('Param')
    Row = NewType("Row", int)

    class Str(sl.Scope[Param, str], str):
        ...

    def parametrized(x: Param) -> Str[Param]:
        return Str(f'{x}')

    def make_float() -> float:
        return 1.5

    pipeline = sl.Pipeline([make_float, parametrized])
    pipeline.set_param_table(sl.ParamTable(Row, {int: [1, 2, 3]}))

    assert pipeline.compute(Str[float]) == Str[float]('1.5')
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.compute(Str[int])
    assert pipeline.compute(sl.Series[Row, Str[int]]) == {
        0: Str[int]('1'),
        1: Str[int]('2'),
        2: Str[int]('3'),
    }


def test_generic_provider_can_depend_on_param_series() -> None:
    Param = TypeVar('Param')
    Row = NewType("Row", int)

    class Str(sl.Scope[Param, str], str):
        ...

    def parametrized_gather(x: sl.Series[Row, Param]) -> Str[Param]:
        return Str(f'{list(x.values())}')

    pipeline = sl.Pipeline([parametrized_gather])
    pipeline.set_param_table(
        sl.ParamTable(Row, {int: [1, 2, 3], float: [1.5, 2.5, 3.5]})
    )

    assert pipeline.compute(Str[int]) == Str[int]('[1, 2, 3]')
    assert pipeline.compute(Str[float]) == Str[float]('[1.5, 2.5, 3.5]')
