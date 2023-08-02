# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from sciline.graph import find_all_paths


def test_find_all_paths() -> None:
    graph = {"D": ["B", "C"], "C": ["A"], "B": ["A"]}
    assert find_all_paths(graph, "D", "A") == [["D", "B", "A"], ["D", "C", "A"]]
    assert find_all_paths(graph, "B", "C") == []
    assert find_all_paths(graph, "B", "A") == [["B", "A"]]
