# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: PYI019
from __future__ import annotations

import itertools
from collections.abc import Callable, Generator, Iterable, Mapping
from types import NoneType
from typing import TYPE_CHECKING, Any, TypeVar, get_args

import cyclebane as cb
import networkx as nx
from cyclebane.node_values import IndexName, IndexValue

from ._provider import ArgSpec, Provider, ToProvider, _bind_free_typevars
from ._utils import key_full_qualname
from .handler import ErrorHandler, HandleAsBuildTimeException
from .typing import Graph, Key

if TYPE_CHECKING:
    import graphviz


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


def _get_typevar_constraints(
    t: TypeVar, over_constraints: dict[TypeVar, frozenset[Key]]
) -> frozenset[Key]:
    """Returns the set of constraints of a TypeVar."""
    if (override := over_constraints.get(t, None)) is not None:
        return override
    if not (constraints := t.__constraints__):
        raise ValueError(
            f"Type variable {t!r} has no constraints. Either constrain the type "
            f"variable in its definition or via the 'constraints' argument of Pipeline."
        )
    return frozenset(constraints)


def _mapping_to_constrained(
    type_vars: set[TypeVar], over_constraints: dict[TypeVar, frozenset[Key]]
) -> Generator[dict[TypeVar, Key], None, None]:
    constraints = [_get_typevar_constraints(t, over_constraints) for t in type_vars]
    for combination in itertools.product(*constraints):
        yield dict(zip(type_vars, combination, strict=True))


def _normalize_custom_constraints(
    constraints: Mapping[TypeVar, Iterable[Key]] | None,
) -> dict[TypeVar, frozenset[Key]]:
    if constraints is None:
        return {}

    normalized = {}
    for key, value in constraints.items():
        types = frozenset(value)
        for ty in types:
            if key.__constraints__ and ty not in key.__constraints__:
                raise ValueError(
                    f"Constraint '{key_full_qualname(ty)}' is not valid for type var "
                    f"'{key_full_qualname(key)}' which supports constraints "
                    f"{tuple(map(key_full_qualname, key.__constraints__))}."
                )
        normalized[key] = types
    return normalized


T = TypeVar('T', bound='DataGraph')


class DataGraph:
    def __init__(
        self,
        providers: None | Iterable[ToProvider | Provider],
        *,
        constraints: Mapping[TypeVar, Iterable[Key]] | None = None,
    ) -> None:
        self._constraints = _normalize_custom_constraints(constraints)
        self._cbgraph = cb.Graph(nx.DiGraph())
        for provider in providers or []:
            self.insert(provider)

    @classmethod
    def _from_cyclebane(cls: type[T], graph: cb.Graph) -> T:
        out = cls([])
        out._cbgraph = graph
        return out

    def copy(self: T) -> T:
        cpy = self._from_cyclebane(self._cbgraph.copy())
        cpy._constraints = self._constraints
        return cpy

    def __copy__(self: T) -> T:
        return self.copy()

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        """Names of the indices (dimensions) of the graph."""
        return self._cbgraph.index_names

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        """Names and values of the indices of the graph."""
        return self._cbgraph.indices

    @property
    def underlying_graph(self) -> nx.DiGraph:
        """The underlying NetworkX graph."""
        return self._cbgraph.graph

    def _get_clean_node(self, key: Key) -> Any:
        """Return node ready for setting value or provider."""
        if key is NoneType:
            raise ValueError('Key must not be None')
        if key in self.underlying_graph:
            self.underlying_graph.remove_edges_from(
                list(self.underlying_graph.in_edges(key))
            )
            self.underlying_graph.nodes[key].pop('value', None)
            self.underlying_graph.nodes[key].pop('provider', None)
            self.underlying_graph.nodes[key].pop('reduce', None)
        else:
            self.underlying_graph.add_node(key)
        return self.underlying_graph.nodes[key]

    def insert(self, provider: ToProvider | Provider, /) -> None:
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
            for bound in _mapping_to_constrained(typevars, self._constraints):
                self.insert(provider.bind_type_vars(bound))
            return
        # Trigger UnboundTypeVar error if any input typevars are not bound
        provider = provider.bind_type_vars({})
        self._get_clean_node(return_type)['provider'] = provider
        for dep in provider.arg_spec.keys():
            self.underlying_graph.add_edge(dep, return_type, key=dep)

    def __setitem__(self, key: Key, value: DataGraph | Any) -> None:
        """
        Provide a concrete value for a type.

        Parameters
        ----------
        key:
            Type to provide a value for.
        value:
            Concrete value to provide.
        """
        # This is a questionable approach: Using MyGeneric[T] as a key will actually
        # not pass mypy [valid-type] checks. What we do on our side is ok, but the
        # calling code is not.
        if typevars := _find_all_typevars(key):
            for bound in _mapping_to_constrained(typevars, self._constraints):
                self[_bind_free_typevars(key, bound)] = value
            return

        # TODO If key is generic, should we support multi-sink case and update all?
        # Would imply that we need the same for __getitem__.
        self._cbgraph[key] = (
            value._cbgraph if isinstance(value, DataGraph) else _as_graph(key, value)
        )

    def __getitem__(self: T, key: Key) -> T:
        """Return the subgraph that computes the given key."""
        return self._from_cyclebane(self._cbgraph[key])

    def map(self: T, node_values: dict[Key, Any]) -> T:
        """Map the graph over given node values.

        Creates a new graph where given nodes and their dependents are duplicated for
        each given value and values are assigned to the given nodes.

        Parameters
        ----------
        node_values:
            Dictionary mapping nodes keys to collections of values.

        Returns
        -------
        :
            A new graph with mapped nodes.
        """
        return self._from_cyclebane(self._cbgraph.map(node_values))

    def reduce(self: T, *, func: Callable[..., Any], **kwargs: Any) -> T:
        """Reduce the outputs of a mapped graph into a single value and provider.

        Parameters
        ----------
        func:
            Function that takes the values to reduce and returns a single value.
            This function is passed as many arguments as there are values to reduce.
        kwargs:
            Forwarded to :meth:`cyclebane.Graph.reduce`.

        Returns
        -------
        :
            A new graph with a new node that depends on all sink nodes of the input
            graph and returns the output of ``func``.
        """
        # Note that the type hints of `func` are not checked here. As we are explicit
        # about the modification, this is in line with __setitem__ which does not
        # perform such checks and allows for using generic reduction functions.
        return self._from_cyclebane(
            self._cbgraph.reduce(attrs={'reduce': func}, **kwargs)
        )

    def to_networkx(self) -> nx.DiGraph:
        return self._cbgraph.to_networkx()

    def visualize_data_graph(self, **kwargs: Any) -> graphviz.Digraph:
        import graphviz

        dot = graphviz.Digraph(strict=True, **kwargs)
        for node in self.underlying_graph.nodes:
            dot.node(str(node), label=str(node), shape='box')
            attrs = self.underlying_graph.nodes[node]
            attrs = '\n'.join(f'{k}={v}' for k, v in attrs.items())
            dot.node(str(node), label=f'{node}\n{attrs}', shape='box')
        for edge in self.underlying_graph.edges:
            key = self.underlying_graph.edges[edge].get('key')
            label = str(key) if key is not None else ''
            dot.edge(str(edge[0]), str(edge[1]), label=label)
        return dot


_no_value = object()


def to_task_graph(
    data_graph: DataGraph, targets: tuple[Key, ...], handler: ErrorHandler | None = None
) -> Graph:
    graph = data_graph.to_networkx()
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
            new_key = dict(zip(orig_keys, input_nodes, strict=True))
            # By using map_keys (instead of creating an ArgSpec from scratch),
            # we automatically preserve what args and kwargs are.
            spec = provider.arg_spec.map_keys(new_key.get)
            if len(spec) != len(input_nodes):
                # This should be caught by __setitem__, but we check here to be safe.
                raise ValueError("Corrupted graph")
            out[key] = Provider(func=provider.func, arg_spec=spec, kind='function')
        elif (func := node.get('reduce')) is not None:
            spec = ArgSpec.from_args(*input_nodes)
            out[key] = Provider(func=func, arg_spec=spec, kind='function')
        else:
            out[key] = handler.handle_unsatisfied_requirement(key)
    return out
