# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Iterable
from types import UnionType
from typing import Any, get_args

import networkx as nx

from sciline.task_graph import TaskGraph

from ._provider import Provider, ToProvider
from .typing import Key

# Refactoring plan
# ----------------
# 1. Build graph directly in init
# 2. Graph is a data graph
#    - Providers are node attrs
#    - Function arg names are edge attrs
#    - Optional/Union means we have multiple edges with same attr.
#      This will be resolved when building a task graph.
#    - Generics must be constrained and are spelled out explicitly.
# 3. Setting values sets a node attr
#    - This may replace a provider, and thus remove incoming edges
# 4. We may want to split the graph building and value setting into two classes,
#    but initially we can keep it as one such that unit tests can be reused.
# 5. Should default values be allowed? Could be stored as edge attr? Sounds brittle.


class DependencyGraph:
    def __init__(self, providers: Iterable[ToProvider | Provider]) -> None:
        self._graph = nx.DiGraph()
        for provider in providers:
            self.add(provider)

    def add(self, provider):
        if not isinstance(provider, Provider):
            provider = Provider.from_function(provider)
        return_type = provider.deduce_key()
        if return_type in self._graph:
            self._graph.remove_edges_from(self._graph.in_edges(return_type))
            self._graph.nodes[return_type].pop('value', None)
        else:
            self._graph.add_node(return_type)
        self._graph.nodes[return_type]['provider'] = provider
        for name, dep in provider.arg_spec.items():
            if isinstance(dep, UnionType):
                for arg in get_args(dep):
                    # Same key for all edges
                    self._graph.add_edge(arg, return_type, key=name)
            else:
                self._graph.add_edge(dep, return_type, key=name)

    def __setitem__(self, key: Key, value: DependencyGraph | Any) -> None:
        if isinstance(value, DependencyGraph):
            # key must be a unique sink node in value
            sink_nodes = [
                n for n in value._graph.nodes if value._graph.out_degree(n) == 0
            ]
            if len(sink_nodes) != 1:
                raise ValueError('Value must have exactly one sink node')
            if key not in sink_nodes:
                raise ValueError('Key must be a sink node in value')
            # TODO
        else:
            self._graph.remove_edges_from(self._graph.in_edges(key))
            self._graph.nodes[key].pop('provider', None)
            self._graph.nodes[key]['value'] = value

    def bind(self, params: dict[Key, Any]) -> DependencyGraph:
        graph = self._graph.copy()
        for key, value in params.items():
            graph.nodes[key]['value'] = value
        out = DependencyGraph([])
        out._graph = graph
        return out

    def build(self, target: Key) -> TaskGraph:
        graph = {}
        for key in self._graph.nodes:
            node = self._graph.nodes[key]
            if (value := node.get('value')) is not None:
                graph[key] = Provider.parameter(value)
            elif (provider := node.get('provider')) is not None:
                # TODO Handle graph modification and optional/union
                graph[key] = provider
        return TaskGraph(graph=graph, targets=target)

    def visualize(
        self, **kwargs: Any
    ) -> graphviz.Digraph:  # type: ignore[name-defined] # noqa: F821
        import graphviz

        dot = graphviz.Digraph(strict=True, **kwargs)
        for node in self._graph.nodes:
            dot.node(str(node), label=str(node), shape='box')
            provider = self._graph.nodes[node].get('provider')
            value = self._graph.nodes[node].get('value')
            if provider:
                dot.node(
                    str(node),
                    label=f'{node}\nprovider={provider.func.__qualname__}',
                    shape='box',
                )
            elif value:
                dot.node(str(node), label=f'{node}\nvalue={value}', shape='box')

        for edge in self._graph.edges:
            dot.edge(
                str(edge[0]), str(edge[1]), label=str(self._graph.edges[edge]['key'])
            )
        return dot
