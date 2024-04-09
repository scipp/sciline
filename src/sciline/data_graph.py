# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import itertools
from collections.abc import Iterable
from types import UnionType
from typing import Any, Generator, TypeVar, Union, get_args, get_origin

import networkx as nx

from sciline.task_graph import TaskGraph

from ._provider import ArgSpec, Provider, ToProvider, _bind_free_typevars
from .handler import UnsatisfiedRequirement
from .scheduler import Scheduler
from .typing import Key
from .util import find_all_typevars, get_typevar_constraints


def _mapping_to_constrained(
    type_vars: set[TypeVar],
) -> Generator[dict[TypeVar, type], None, None]:
    constraints = [get_typevar_constraints(t) for t in type_vars]
    if any(len(c) == 0 for c in constraints):
        raise ValueError('Typevars must have constraints')
    for combination in itertools.product(*constraints):
        yield dict(zip(type_vars, combination))


def _is_multiple_keys(keys: type | Iterable[type]) -> bool:
    # Cannot simply use isinstance(keys, Iterable) because that is True for
    # generic aliases of iterable types, e.g.,
    #
    # class Str(sl.Scope[Param, str], str): ...
    # keys = Str[int]
    #
    # And isinstance(keys, type) does not work on its own because
    # it is False for the above type.
    return (
        not isinstance(keys, type) and not get_args(keys) and isinstance(keys, Iterable)
    )


def _prune_unsatisfied(graph: nx.DiGraph) -> nx.DiGraph:
    """Remove nodes without value or provider."""
    # TODO This prunes only source, but we may need to prune more, until we reach
    # an optional/union node.
    graph = graph.copy()
    for node in list(graph.nodes):
        if not graph.nodes[node].get('value') and not graph.nodes[node].get('provider'):
            # TODO This was a way of handling missing inputs, but it interferes with
            # the support for Optional and Union. Need to think of a better way.
            #
            # descendants = nx.descendants(graph, node)
            # out_edges = list(graph.out_edges(node, data=True))
            # out_keys = [edge[2].get('key') for edge in out_edges]
            # for out_key, descendant in zip(out_keys, descendants):
            #     in_edges = list(graph.in_edges(descendant, data=True))
            #     in_keys = [edge[2].get('key') for edge in in_edges]
            #     # TODO If the input has a default we would not want to raise?
            #     # if in_keys.count(out_key) == 1:
            #     #    raise UnsatisfiedRequirement(f'No provider for type {out_key}')
            graph.remove_node(node)
    return graph


class DataGraph:
    def __init__(self, providers: None | Iterable[ToProvider | Provider]) -> None:
        self._graph = nx.DiGraph()
        for provider in providers or []:
            self.add(provider)

    def _get_clean_node(self, key: Key) -> Any:
        """Return node ready for setting value or provider."""
        if key in self._graph:
            self._graph.remove_edges_from(list(self._graph.in_edges(key)))
            self._graph.nodes[key].pop('value', None)
            self._graph.nodes[key].pop('provider', None)
        else:
            self._graph.add_node(key)
        return self._graph.nodes[key]

    def add(self, provider):
        if not isinstance(provider, Provider):
            provider = Provider.from_function(provider)
        return_type = provider.deduce_key()
        if typevars := find_all_typevars(return_type):
            for bound in _mapping_to_constrained(typevars):
                self.add(provider.bind_type_vars(bound))
            return
        self._get_clean_node(return_type)['provider'] = provider
        for dep in provider.arg_spec.keys():
            if isinstance(dep, UnionType) or get_origin(dep) == Union:
                for arg in get_args(dep):
                    # Same key for all edges
                    self._graph.add_edge(arg, return_type, key=dep)
            else:
                self._graph.add_edge(dep, return_type, key=dep)

    def __setitem__(self, key: Key, value: DataGraph | Any) -> None:
        if typevars := find_all_typevars(key):
            for bound in _mapping_to_constrained(typevars):
                self[_bind_free_typevars(key, bound)] = value
            return
        if isinstance(value, DataGraph):
            # TODO If key is generic, should we support multi-sink case and update all?
            # Would imply that we need the same for __getitem__.
            # key must be a unique sink node in value
            sinks = [n for n in value._graph.nodes if value._graph.out_degree(n) == 0]
            if len(sinks) != 1:
                raise ValueError('Value must have exactly one sink node')
            if key not in sinks:
                raise ValueError('Key must be a sink node in value')
            _ = self._get_clean_node(key)
            # TODO Conflict handling?
            self._graph = nx.compose(self._graph, value._graph)
        else:
            self._get_clean_node(key)['value'] = value

    def to_cyclebane(
        self,
    ) -> cyclebane.Graph:  # type: ignore[no-untyped-def] # noqa: F821
        import cyclebane as cb

        return cb.Graph(self._graph)

    def copy(self) -> DataGraph:
        out = self.__class__([])
        out._graph = self._graph.copy()
        return out

    def bind(self, params: dict[Key, Any]) -> DataGraph:
        out = self.copy()
        for key, value in params.items():
            out[key] = value
        return out

    def build(self, target: Key, scheduler: None | Scheduler = None) -> TaskGraph:
        return to_task_graph(self._graph, target=target, scheduler=scheduler)

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


def to_task_graph(
    graph: nx.DiGraph, target: Key, scheduler: None | Scheduler = None
) -> TaskGraph:
    if _is_multiple_keys(target):
        targets = tuple(target)  # type: ignore[arg-type]
    else:
        targets = (target,)
    ancestors = list(targets)
    for node in targets:
        if node not in graph:
            raise UnsatisfiedRequirement(f'No provider for type {node}')
        ancestors.extend(nx.ancestors(graph, node))
    graph = graph.subgraph(set(ancestors))
    graph = _prune_unsatisfied(graph)
    if any(node not in graph for node in targets):
        raise UnsatisfiedRequirement(f'No provider for type {target}')
    out = {}

    for key in graph.nodes:
        node = graph.nodes[key]
        input_nodes = list(graph.predecessors(key))
        input_edges = list(graph.in_edges(key, data=True))
        orig_keys = [edge[2].get('key', None) for edge in input_edges]
        if (value := node.get('value')) is not None:
            out[key] = Provider.parameter(value)
        elif (provider := node.get('provider')) is not None:
            if not isinstance(provider, Provider):
                # This happens when using cyclebane
                new_spec = ArgSpec.from_args(*input_nodes)
                out[key] = Provider(func=provider, arg_spec=new_spec, kind='function')
            else:
                new_key = {orig_key: n for n, orig_key in zip(input_nodes, orig_keys)}
                spec = provider.arg_spec.map_keys(new_key.get)
                # TODO I added this for optional/default/missing handling but it breaks
                # map-reduce ops
                # spec._args = {k: v for k, v in spec._args.items() if v in new_key}
                # TODO also kwargs
                # TODO Raise if no default?
                out[key] = Provider(func=provider.func, arg_spec=spec, kind='function')
        else:
            raise ValueError('Node must have a provider or a value')
    return TaskGraph(graph=out, targets=target, scheduler=scheduler)
