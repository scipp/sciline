# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Iterable
from types import UnionType
from typing import Any, Union, get_args, get_origin

import networkx as nx

from sciline.task_graph import TaskGraph

from ._provider import Provider, ToProvider
from .typing import Key


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
        for dep in provider.arg_spec.keys():
            if isinstance(dep, UnionType) or get_origin(dep) == Union:
                for arg in get_args(dep):
                    # Same key for all edges
                    self._graph.add_edge(arg, return_type, key=dep)
            else:
                self._graph.add_edge(dep, return_type, key=dep)

    def __setitem__(self, key: Key, value: DependencyGraph | Any) -> None:
        if isinstance(value, DependencyGraph):
            # key must be a unique sink node in value
            sinks = [n for n in value._graph.nodes if value._graph.out_degree(n) == 0]
            if len(sinks) != 1:
                raise ValueError('Value must have exactly one sink node')
            if key not in sinks:
                raise ValueError('Key must be a sink node in value')
            self._graph.remove_edges_from(self._graph.in_edges(key))
            self._graph.nodes[key].pop('value', None)
            self._graph.nodes[key].pop('provider', None)
            # TODO Conflict handling?
            self._graph = nx.compose(self._graph, value._graph)
        else:
            self._graph.remove_edges_from(self._graph.in_edges(key))
            self._graph.nodes[key].pop('provider', None)
            self._graph.nodes[key]['value'] = value

    def bind(self, params: dict[Key, Any]) -> DependencyGraph:
        out = DependencyGraph([])
        out._graph = self._graph.copy()
        for key, value in params.items():
            out[key] = value
        return out

    def _prune_unsatisfied(self) -> nx.DiGraph:
        """Remove nodes without value or provider."""
        # TODO This prunes only source, but we may need to prune more, until we reach
        # an optional/union node.
        graph = self._graph.copy()
        for node in list(graph.nodes):
            if not graph.nodes[node].get('value') and not graph.nodes[node].get(
                'provider'
            ):
                graph.remove_node(node)
        return graph

    def build(self, target: Key) -> TaskGraph:
        graph = self._prune_unsatisfied()
        out = {}

        for key in graph.nodes:
            node = graph.nodes[key]
            input_nodes = list(graph.predecessors(key))
            input_edges = list(graph.in_edges(key, data=True))
            orig_keys = [edge[2].get('key', None) for edge in input_edges]
            if (value := node.get('value')) is not None:
                out[key] = Provider.parameter(value)
            elif (provider := node.get('provider')) is not None:
                new_key = {orig_key: n for n, orig_key in zip(input_nodes, orig_keys)}
                arg_spec = provider.arg_spec.map_keys(new_key.get)
                out[key] = Provider(
                    func=provider.func, arg_spec=arg_spec, kind='function'
                )
            else:
                raise ValueError('Node must have a provider or a value')
        return TaskGraph(graph=out, targets=target)

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
