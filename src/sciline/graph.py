# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import TypeVar

import networkx as nx

from sciline import HandleAsComputeTimeException, Pipeline
from sciline.typing import get_optional
from sciline.visualize import _format_type as format_type

T = TypeVar('T')


class Graph:
    def __init__(self, p: Pipeline, keys=None):
        self._graph = self._to_networkx(p, keys=keys)

    def _to_networkx(self, p: Pipeline, keys=None) -> Graph:
        keys = keys or []
        tg = p.get(
            list(p._providers.keys()) + keys,
            # list(p._providers.keys()) + list(p._subproviders.keys()),
            handler=HandleAsComputeTimeException(),
        )
        g = nx.DiGraph()
        for res, (provider, args) in tg._graph.items():
            # When a provider depends on Optional[T], this will result in an item
            # with res=Optional[T]. However, we also directly request T above, so
            # there is some duplication that we need to avoid.
            res = get_optional(res) or res
            result_name = format_type(res).name
            provider_name = provider.__qualname__

            # If result_name ends in an [...] part, append that to provider_name
            if result_name.endswith(']'):
                i = result_name.rfind('[')
                provider_name = provider_name + result_name[i:]
            is_input_data = provider_name.startswith('Pipeline.__setitem__.')
            g.add_node(
                hash(res), label=result_name, type='input' if is_input_data else 'data'
            )
            if is_input_data:
                continue
            if provider_name.startswith(
                'HandleAsComputeTimeException.handle_unsatisfied_requirement.'
            ):
                continue
            # TODO should follow ambiguous providers? Would be useful for debugging,
            # but how? Logic in Pipeline cannot
            g.add_node(hash(provider_name), label=provider_name, type='provider')
            g.add_edge(hash(provider_name), hash(res))
            for arg in args:
                arg = get_optional(arg) or arg
                g.add_edge(hash(arg), hash(provider_name))
        return g

    def unsatisfied_requirement_nodes(self) -> list[int]:
        """Return 'data' nodes that have no incoming edges."""
        return [
            node
            for node in self._graph.nodes()
            if self._graph.in_degree(node) == 0
            and self._graph.nodes[node]['type'] == 'data'
        ]

    def unsatisfied_requirement_names(self) -> list[str]:
        """Return names of 'data' nodes that have no incoming edges."""
        return [
            self._graph.nodes[node]['label']
            for node in self.unsatisfied_requirement_nodes()
        ]

    def output_nodes(self) -> list[str]:
        """Return 'data' nodes that have no outgoing edges."""
        return [
            node
            for node in self._graph.nodes()
            if self._graph.out_degree(node) == 0
            and self._graph.nodes[node]['type'] == 'data'
        ]

    def output_names(self) -> list[str]:
        """Return names of 'data' nodes that have no outgoing edges."""
        return [self._graph.nodes[node]['label'] for node in self.output_nodes()]

    def visualize(self):
        """Visualize the graph using graphviz."""
        agraph = nx.nx_agraph.to_agraph(self._graph)

        for node in agraph.nodes():
            if node.attr['type'] in ['data', 'input']:
                node.attr['shape'] = 'rectangle'

        for node in self.unsatisfied_requirement_nodes():
            agraph.get_node(node).attr['color'] = 'red'
            agraph.get_node(node).attr['fontcolor'] = 'red'

        agraph.layout('dot')
        return agraph
