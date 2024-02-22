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
    renamed_nodes = {
        node: node + (index,) if isinstance(node, tuple) else (node, index)
        for node in successors
    }
    renamed_nodes[node] = index
    return nx.relabel_nodes(graph, renamed_nodes, copy=True)


class TaskGraph2:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.labels: dict[Hashable, list[Hashable]] = {}

    def map(self, node: Hashable, values: list[Hashable]) -> TaskGraph2:
        """For every value, create a new graph with all successors renamed, merge all
        resulting graphs."""
        # TODO Should allow mapping only for inputs nodes, not for intermediate nodes
        graphs = [
            rename_successors(self.graph, node, index=(node, i))
            for i in range(len(values))
        ]
        graph = TaskGraph2(nx.compose_all(graphs))
        graph.labels = {**self.labels}
        graph.labels[node] = (node, values)
        return graph

    def __getitem__(self, sel: tuple[str, int]) -> TaskGraph2:
        """
        Return a new graph, essentially undoing the effect of `map`.

        Remove any node that has (key, i) for i != index, and remove the key from the
        labels.
        """
        key, index = sel
        graph = self.graph.copy()
        drop = []
        remain = []
        for node in self.graph.nodes:
            if isinstance(node, tuple):
                name, *indices = node
                if name == key and isinstance(indices[0], int):
                    if indices[0] == index:
                        remain.append(node)
                    else:
                        drop.append(node)
                elif name != key:
                    for dim, i in indices:
                        if dim == key:
                            if i == index:
                                remain.append(node)
                            else:
                                drop.append(node)
                            break
        for node in drop:
            if node[1] != index:
                graph.remove_node(node)
        # TODO remove index from remaining nodes, replace root by label
        print(remain)
        graph = TaskGraph2(graph)
        graph.labels = {**self.labels}
        del graph.labels[key]
        return graph

    def reduce(self, key: str, index: Hashable, func: str) -> TaskGraph2:
        """Add edges from all nodes (key, index) to new node func."""
        nodes = [node for node in self.graph.nodes if node[0] == key]
        new_node = func
        graph = self.graph.copy()
        for node in nodes:
            # Node looks like ('v', ('x', 2), ('y', 11))
            _, *indices = node
            # Remove the old index, e.g., ('x', 2) if index is 'x'
            indices = [i for i in indices if i[0] != index]
            new_node = (func, *indices)
            graph.add_edge(node, new_node)
        graph = TaskGraph2(graph)
        for name, labels in self.labels.items():
            if labels[0] != index:
                graph.labels[name] = labels
        return graph

    def groupby(self, key: str, index: str, reduce: str) -> TaskGraph2:
        """
        Similar to reduce, but group nodes by label.

        Add edges from all nodes (key, index) to new node (func, label), for every
        label.  The label is given by `labels[index]`.
        """
        nodes = [node for node in self.graph.nodes if node[0] == key]
        orig_index, labels = self.labels[index]
        sorted_unique_labels = sorted(set(labels))
        graph = self.graph.copy()
        for node in nodes:
            # Node looks like (key, (orig_index, 2), ('y', 11))
            # We want to add an edge to (reduce, (index, label), ('y', 11))
            _, *indices = node
            orig_pos = [i[1] for i in indices if i[0] == orig_index][0]
            orig_label = labels[orig_pos]
            indices = [i for i in indices if i[0] != orig_index]
            label = sorted_unique_labels.index(orig_label)
            graph.add_edge(node, (reduce, (index, label), *indices))
        graph = TaskGraph2(graph)
        for name, labels in self.labels.items():
            if labels[0] != orig_index:
                graph.labels[name] = labels
        graph.labels[index] = (index, sorted_unique_labels)
        return graph

    def _repr_html_(self):
        from IPython.display import display
        from networkx.drawing.nx_agraph import to_agraph

        A = to_agraph(self.graph)
        A.layout('dot')
        display(A)
