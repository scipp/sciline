# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Hashable

import networkx as nx


def rename_successors(graph: nx.DiGraph, node: str, index: Hashable) -> nx.DiGraph:
    """Replace 'node' and all its successors with (node, suffix), and update all edges
    accordingly."""
    successors = nx.dfs_successors(graph, node)
    # Get set of all successors
    successors = set(
        successor for successors in successors.values() for successor in successors
    )
    renamed_nodes = {node: (node, index) for node in successors}
    renamed_nodes[node] = index
    return nx.relabel_nodes(graph, renamed_nodes, copy=True)


class TaskGraph2:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.labels: dict[Hashable, list[Hashable]] = {}

    def map(self, node: Hashable, values: list[Hashable]) -> TaskGraph2:
        """For every value, create a new graph with all successors renamed, merge all
        resulting graphs."""
        graphs = [
            rename_successors(self.graph, node, index=(node, value)) for value in values
        ]
        graph = TaskGraph2(nx.compose_all(graphs))
        graph.labels = {**self.labels}
        graph.labels[node] = values
        return graph

    def reduce(self, key: str, func: str) -> TaskGraph2:
        """Add edges from all nodes (key, index) to new node func."""
        nodes = [node for node in self.graph.nodes if node[0] == key]
        new_node = func
        graph = self.graph.copy()
        for node in nodes:
            graph.add_edge(node, new_node)
        return TaskGraph2(graph)

    def groupby(self, key: str, label: str, reduce: str) -> TaskGraph2:
        """
        Similar to reduce, but group nodes by label.

        Add edges from all nodes (key, index) to new node (func, label), for every
        label.  The label is given by `labels[index]`.
        """
        nodes = [node for node in self.graph.nodes if node[0] == key]
        index, labels = self.labels[label]
        labels = dict(zip(self.labels[index], labels))
        new_nodes = {label: (reduce, label) for label in labels.values()}
        graph = self.graph.copy()
        for node in nodes:
            label = labels[node[1][1]]
            graph.add_edge(node, new_nodes[label])
        return TaskGraph2(graph)

    def _repr_html_(self):
        from IPython.display import display
        from networkx.drawing.nx_agraph import to_agraph

        A = to_agraph(self.graph)
        A.layout('dot')
        display(A)
