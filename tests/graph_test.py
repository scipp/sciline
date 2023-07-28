# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

from sciline.graph import find_path, find_unique_path


def test_find_path():
    graph = {"D": ["B", "C"], "C": ["A"], "B": ["A"]}
    assert find_path(graph, "D", "A") == ["D", "B", "A"]


def test_find_unique_path():
    graph = {"D": ["B", "C"], "C": ["A"], "B": ["A"]}
    with pytest.raises(RuntimeError):
        find_unique_path(graph, "D", "A")
    graph = {"D": ["B", "C"], "C": ["A"], "B": ["aux"]}
    assert find_unique_path(graph, "D", "A") == ["D", "C", "A"]
