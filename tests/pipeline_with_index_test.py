# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline as sl
from sciline.pipeline import Item, Label


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


def test_can_gather_index() -> None:
    Sum = NewType("Sum", float)
    Name = NewType("Name", str)

    def gather(x: sl.Map[Name, float]) -> Sum:
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

    def gather_zip(x: sl.Map[Run, Str], y: sl.Map[Run, int]) -> Sum:
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
        x: sl.Map[Row, float],
    ) -> Sum:
        return Sum(sum(x.values()))

    def join(x: Param1, y: Param2) -> float:
        return x / y

    pl = sl.Pipeline([gather, join])
    pl.set_param_table(sl.ParamTable(Row, {Param1: [1, 4, 9], Param2: [1, 2, 3]}))

    assert pl.compute(Sum) == 6
