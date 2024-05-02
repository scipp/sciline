# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import itertools
from collections.abc import Iterable
from types import NoneType
from typing import Any, Callable, Generator, Optional, TypeVar, Union, get_args

import cyclebane as cb
import networkx as nx

from ._provider import ArgSpec, Provider, ToProvider, _bind_free_typevars
from .handler import ErrorHandler, HandleAsBuildTimeException
from .typing import Graph, Key


def _as_graph(key: Key, value: Any) -> cb.Graph:
    """Create a cyclebane.Graph with a single value."""
    graph = nx.DiGraph()
    graph.add_node(key, value=value)
    return cb.Graph(graph)


def _find_all_typevars(t: type | TypeVar) -> set[TypeVar]:
    """Returns the set of all TypeVars in a type expression."""
    if isinstance(t, TypeVar):
        return {t}
    return set(itertools.chain(*map(_find_all_typevars, get_args(t))))


def _get_typevar_constraints(t: TypeVar) -> set[type]:
    """Returns the set of constraints of a TypeVar."""
    return set(t.__constraints__)


def _mapping_to_constrained(
    type_vars: set[TypeVar],
) -> Generator[dict[TypeVar, type], None, None]:
    constraints = [_get_typevar_constraints(t) for t in type_vars]
    if any(len(c) == 0 for c in constraints):
        raise ValueError('Typevars must have constraints')
    for combination in itertools.product(*constraints):
        yield dict(zip(type_vars, combination))


T = TypeVar('T', bound='DataGraph')


class DataGraph:
    def __init__(self, providers: None | Iterable[ToProvider | Provider]) -> None:
        self._cbgraph = cb.Graph(nx.DiGraph())
        for provider in providers or []:
            self.insert(provider)

    @classmethod
    def from_cyclebane(cls: type[T], graph: cb.Graph) -> T:
        out = cls([])
        out._cbgraph = graph
        return out

    def copy(self: T) -> T:
        return self.from_cyclebane(self._cbgraph.copy())

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

    def insert(self, provider: Union[ToProvider, Provider], /) -> None:
        """
        Insert a callable into the graph that provides its return value.

        Parameters
        ----------
        provider:
            Either a callable that provides its return value. Its arguments
            and return value must be annotated with type hints.
            Or a ``Provider`` object that has been constructed from such a callable.
        """
        if not isinstance(provider, Provider):
            provider = Provider.from_function(provider)
        return_type = provider.deduce_key()
        if typevars := _find_all_typevars(return_type):
            for bound in _mapping_to_constrained(typevars):
                self.insert(provider.bind_type_vars(bound))
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
        # This is a questionable approach: Using MyGeneric[T] as a key will actually
        # not pass mypy [valid-type] checks. What we do on our side is ok, but the
        # calling code is not.
        if typevars := _find_all_typevars(key):
            for bound in _mapping_to_constrained(typevars):
                self[_bind_free_typevars(key, bound)] = value
            return

        # TODO If key is generic, should we support multi-sink case and update all?
        # Would imply that we need the same for __getitem__.
        self._cbgraph[key] = (
            value._cbgraph if isinstance(value, DataGraph) else _as_graph(key, value)
        )

    def __getitem__(self, key: Key) -> DataGraph:
        return self.from_cyclebane(self._cbgraph[key])

    def map(self, node_values: dict[Key, Any]) -> DataGraph:
        return self.from_cyclebane(self._cbgraph.map(node_values))

    def reduce(self, *, func: Callable[..., Any], **kwargs: Any) -> DataGraph:
        # Note that the type hints of `func` are not checked here. As we are explicit
        # about the modification, this is in line with __setitem__ which does not
        # perform such checks and allows for using generic reduction functions.
        return self.from_cyclebane(
            self._cbgraph.reduce(attrs={'reduce': func}, **kwargs)
        )

    def build(
        self, targets: tuple[Key, ...], handler: Optional[ErrorHandler] = None
    ) -> Graph:
        return _to_task_graph(
            self._cbgraph.to_networkx(), targets=targets, handler=handler
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


def _to_task_graph(
    graph: nx.DiGraph, targets: tuple[Key, ...], handler: Optional[ErrorHandler] = None
) -> Graph:
    handler = handler or HandleAsBuildTimeException()
    ancestors = list(targets)
    for target in targets:
        if target not in graph:
            handler.handle_unsatisfied_requirement(target)
        ancestors.extend(nx.ancestors(graph, target))
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
            if len(spec) != len(input_nodes):
                # This should be caught by __setitem__, but we check here to be safe.
                raise ValueError("Corrupted graph")
            # TODO also kwargs
            out[key] = Provider(func=provider.func, arg_spec=spec, kind='function')
        elif (func := node.get('reduce')) is not None:
            spec = ArgSpec.from_args(*input_nodes)
            out[key] = Provider(func=func, arg_spec=spec, kind='function')
        else:
            out[key] = handler.handle_unsatisfied_requirement(key)
    return out
