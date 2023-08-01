# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline as sl
from sciline.pipeline import Label


def test_set_index_sets_up_providers_with_indexable_instances() -> None:
    pl = sl.Pipeline()
    pl.set_index(float, [1.0, 2.0, 3.0])
    assert pl.compute(Label(float, 0)) == 1.0
    assert pl.compute(Label(float, 1)) == 2.0
    assert pl.compute(Label(float, 2)) == 3.0


def test_can_depend_on_index_elements() -> None:
    def use_index_elem(x: Label(float, 1)) -> int:
        return int(x)

    pl = sl.Pipeline([use_index_elem])
    pl.set_index(float, [1.0, 2.0, 3.0])
    assert pl.compute(int) == 2


def test_can_gather_index() -> None:
    Sum = NewType("Sum", float)

    def gather(x: sl.Map[str, float]) -> Sum:
        print(list(x.items()))
        return Sum(sum(x.values()))

    def make_float(x: str) -> float:
        return float(x)

    pl = sl.Pipeline([gather, make_float])
    pl.set_index(str, ["1.0", "2.0", "3.0"])
    graph = pl.build(Sum)
    for key, value in graph.items():
        print(key, value)
    assert pl.compute(Sum) == 6.0
