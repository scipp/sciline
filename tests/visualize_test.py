# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
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
