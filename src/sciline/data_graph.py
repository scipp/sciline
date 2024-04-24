# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import itertools
from collections.abc import Iterable
from types import NoneType
from typing import Any, Generator, Optional, TypeVar, get_args

import cyclebane as cb
import networkx as nx

from sciline.task_graph import TaskGraph

from ._provider import ArgSpec, Provider, ToProvider, _bind_free_typevars
from .handler import ErrorHandler, HandleAsBuildTimeException
from .scheduler import Scheduler
from .typing import Key


def find_all_typevars(t: type | TypeVar) -> set[TypeVar]:
    """Returns the set of all TypeVars in a type expression."""
    if isinstance(t, TypeVar):
        return {t}
    return set(itertools.chain(*map(find_all_typevars, get_args(t))))


def get_typevar_constraints(t: TypeVar) -> set[type]:
    """Returns the set of constraints of a TypeVar."""
    return set(t.__constraints__)


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


class DataGraph:
    def __init__(self, providers: None | Iterable[ToProvider | Provider]) -> None:
        self._cbgraph = cb.Graph(nx.DiGraph())
        for provider in providers or []:
            self.add(provider)

    @classmethod
    def from_cyclebane(cls, graph: cb.Graph) -> DataGraph:
        out = cls([])
        out._cbgraph = graph
        return out

    @property
    def _graph(self) -> nx.DiGraph:
        return self._cbgraph.graph

    def _get_clean_node(self, key: Key) -> Any:
        """Return node ready for setting value or provider."""
        if key is NoneType:
            raise ValueError('Key must not be None')
        if key in self._graph:
            self._graph.remove_edges_from(list(self._graph.in_edges(key)))
            self._graph.nodes[key].pop('value', None)
            self._graph.nodes[key].pop('provider', None)
            self._graph.nodes[key].pop('reduce', None)
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
        # Trigger UnboundTypeVar error if any input typevars are not bound
        provider = provider.bind_type_vars({})
        self._get_clean_node(return_type)['provider'] = provider
        for dep in provider.arg_spec.keys():
            self._graph.add_edge(dep, return_type, key=dep)

    def __setitem__(self, key: Key, value: DataGraph | Any) -> None:
        """
        Provide a concrete value for a type.

        Parameters
        ----------
        key:
            Type to provide a value for.
        param:
            Concrete value to provide.
        """
        if typevars := find_all_typevars(key):
            for bound in _mapping_to_constrained(typevars):
                self[_bind_free_typevars(key, bound)] = value
            return
        if isinstance(value, DataGraph):
            # TODO If key is generic, should we support multi-sink case and update all?
            # Would imply that we need the same for __getitem__.
            # key must be a unique sink node in value
            self._cbgraph[key] = value._cbgraph
        else:
            self._get_clean_node(key)['value'] = value

    def __getitem__(self, key: Key) -> DataGraph:
        graph = self._cbgraph[key]
        return self.from_cyclebane(graph)

    def map(self, *args, **kwargs) -> DataGraph:
        graph = self._cbgraph.map(*args, **kwargs)
        return self.from_cyclebane(graph)

    def reduce(self, *, func, **kwargs) -> DataGraph:
        # Note that the type hints of `func` are not checked here. As we are explicit
        # about the modification, this is in line with __setitem__ which does not
        # perform such checks and allows for using generic reduction functions.
        graph = self._cbgraph.reduce(attrs={'reduce': func}, **kwargs)
        return self.from_cyclebane(graph)

    def copy(self) -> DataGraph:
        return self.from_cyclebane(self._cbgraph.copy())

    def bind(self, params: dict[Key, Any]) -> DataGraph:
        out = self.copy()
        for key, value in params.items():
            out[key] = value
        return out

    def build(
        self,
        target: Key,
        scheduler: None | Scheduler = None,
        handler: Optional[ErrorHandler] = None,
    ) -> TaskGraph:
        return to_task_graph(
            self._cbgraph.to_networkx(),
            target=target,
            scheduler=scheduler,
            handler=handler,
        )

    def visualize_data_graph(
        self, **kwargs: Any
    ) -> graphviz.Digraph:  # type: ignore[name-defined] # noqa: F821
        import graphviz

        dot = graphviz.Digraph(strict=True, **kwargs)
        for node in self._graph.nodes:
            dot.node(str(node), label=str(node), shape='box')
            attrs = self._graph.nodes[node]
            attrs = '\n'.join(f'{k}={v}' for k, v in attrs.items())
            dot.node(str(node), label=f'{node}\n{attrs}', shape='box')
        for edge in self._graph.edges:
            key = self._graph.edges[edge].get('key')
            label = str(key) if key is not None else ''
            dot.edge(str(edge[0]), str(edge[1]), label=label)
        return dot


_no_value = object()


def to_task_graph(
    graph: nx.DiGraph,
    target: Key,
    scheduler: None | Scheduler = None,
    handler: Optional[ErrorHandler] = None,
) -> TaskGraph:
    handler = handler or HandleAsBuildTimeException()
    if multi := _is_multiple_keys(target):
        targets = tuple(target)  # type: ignore[arg-type]
    else:
        targets = (target,)
    ancestors = list(targets)
    for node in targets:
        if node not in graph:
            handler.handle_unsatisfied_requirement(node)
        ancestors.extend(nx.ancestors(graph, node))
    graph = graph.subgraph(set(ancestors))
    out = {}

    for key in graph.nodes:
        node = graph.nodes[key]
        input_nodes = list(graph.predecessors(key))
        input_edges = list(graph.in_edges(key, data=True))
        orig_keys = [edge[2].get('key', None) for edge in input_edges]
        if (value := node.get('value', _no_value)) is not _no_value:
            out[key] = Provider.parameter(value)
        elif (provider := node.get('provider')) is not None:
            new_key = dict(zip(orig_keys, input_nodes))
            spec = provider.arg_spec.map_keys(new_key.get)
            # TODO also kwargs
            out[key] = Provider(func=provider.func, arg_spec=spec, kind='function')
        elif (func := node.get('reduce')) is not None:
            spec = ArgSpec.from_args(*input_nodes)
            out[key] = Provider(func=func, arg_spec=spec, kind='function')
        else:
            out[key] = handler.handle_unsatisfied_requirement(key)
    return TaskGraph(
        graph=out,
        targets=targets if multi else target,
        scheduler=scheduler,
    )
