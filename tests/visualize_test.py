# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Generic, TypeVar

import sciline as sl
from sciline.visualize import to_graphviz


def test_can_visualize_graph_with_cycle() -> None:
    def int_to_float(x: int) -> float:
        return float(x)

    def float_to_int(x: float) -> int:
        return int(x)

    pipeline = sl.Pipeline([int_to_float, float_to_int])
    graph = pipeline.build(int)
    to_graphviz(graph)


def test_generic_types_formatted_without_prefixes() -> None:
    T = TypeVar('T')

    class A(sl.Scope[T, int], int):
        pass

    class B(Generic[T]):
        pass

    class SubA(A[T]):
        pass

    assert sl.visualize._format_type(A[float]) == 'A[float]'
    assert sl.visualize._format_type(SubA[float]) == 'SubA[float]'
    assert sl.visualize._format_type(B[float]) == 'B[float]'
