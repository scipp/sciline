# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Dict, List, Tuple

from graphviz import Digraph


def make_graph(graph: Dict[str, Tuple[List[str], str]]) -> Digraph:
    """
    Convert output of :py:class:`sciline.Pipeline.get_graph` to a graphviz graph.

    Parameters
    ----------
    graph:
        Output of :py:class:`sciline.Pipeline.get_graph`.
    """
    dot = Digraph(strict=True)
    for p, (args, ret) in graph.items():
        dot.node(p, p, shape='ellipse')
        for arg in args:
            dot.node(arg, arg, shape='rectangle')
            dot.edge(arg, p)
        dot.node(ret, ret, shape='rectangle')
        dot.edge(p, ret)
    return dot
