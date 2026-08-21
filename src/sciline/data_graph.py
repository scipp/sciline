# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: PYI019
from __future__ import annotations

import itertools
from collections.abc import Callable, Generator, Iterable, Mapping
from types import NoneType
from typing import TYPE_CHECKING, Any, TypeVar

import cyclebane as cb
import networkx as nx
from cyclebane.node_values import IndexName, IndexValue

from ._provider import (
    ArgSpec,
    Provider,
    ToProvider,
    UnboundTypeVar,
    _bind_free_typevars,
)
from ._unification import find_all_typevars, forward_bindings, match_return
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

_providing_attrs = frozenset(('value', 'provider', 'reduce'))


class DataGraph:
    def __init__(
        self,
        providers: None | Iterable[ToProvider | Provider],
        *,
        constraints: Mapping[TypeVar, Iterable[Key]] | None = None,
    ) -> None:
        self._constraints = _normalize_custom_constraints(constraints)
        self._templates: list[Provider] = []
        self._cbgraph = cb.Graph(nx.DiGraph())
        for provider in providers or []:
            self.insert(provider)

    def _from_cyclebane(self: T, graph: cb.Graph) -> T:
        out = type(self)([])
        out._cbgraph = graph
        out._constraints = self._constraints
        out._templates = list(self._templates)
        return out

    def copy(self: T) -> T:
        return self._from_cyclebane(self._cbgraph.copy())

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
        if typevars := find_all_typevars(return_type):
            if all(t.__constraints__ or t in self._constraints for t in typevars):
                for bound in _mapping_to_constrained(typevars, self._constraints):
                    self.insert(provider.bind_type_vars(bound))
            else:
                self._register_template(provider, typevars)
            return
        # Trigger UnboundTypeVar error if any input typevars are not bound
        provider = provider.bind_type_vars({})
        self._get_clean_node(return_type)['provider'] = provider
        for dep in provider.arg_spec.keys():
            self.underlying_graph.add_edge(dep, return_type, key=dep)

    def _register_template(self, provider: Provider, typevars: set[TypeVar]) -> None:
        """Store a generic provider for on-demand instantiation.

        Providers with unconstrained type variables cannot be expanded eagerly.
        They are instantiated later by unifying their type patterns with the
        concrete keys that appear in the graph, see :py:meth:`_instantiate_backward`
        and :py:meth:`_instantiate_forward`.
        """
        return_type = provider.deduce_key()
        if isinstance(return_type, TypeVar):
            raise ValueError(
                f"Provider {provider} returns a bare unconstrained type variable "
                f"{return_type!r}, which would match any requested key. Use a "
                "generic class as return type or constrain the type variable."
            )
        arg_typevars: set[TypeVar] = set()
        for arg in provider.arg_spec.keys():
            arg_typevars |= find_all_typevars(arg)
        if unbound := arg_typevars - typevars:
            raise UnboundTypeVar(
                f"Provider {provider} has type variables {unbound} in its "
                "arguments that do not appear in its return type."
            )
        # Mirror the replacement semantics of inserting a concrete provider twice.
        self._templates = [t for t in self._templates if t != provider]
        self._templates.append(provider)

    def _satisfied(self, key: Key) -> bool:
        graph = self.underlying_graph
        return key in graph and bool(graph.nodes[key].keys() & _providing_attrs)

    def _instantiate_backward(self, keys: Iterable[Key]) -> None:
        """Instantiate templates for demanded keys and their dependencies."""
        stack = list(keys)
        seen = set()
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            if not self._satisfied(key):
                # Iterate in reverse so that the latest matching template wins,
                # mirroring the replacement semantics of concrete providers.
                for template in reversed(self._templates):
                    if (provider := match_return(template, key)) is not None:
                        self.insert(provider)
                        break
            if key in self.underlying_graph:
                stack.extend(self.underlying_graph.predecessors(key))

    def _instantiate_forward(self, extra_keys: Iterable[Key] = ()) -> None:
        """Instantiate templates whose arguments unify with concrete keys.

        Runs to a fixed point since instantiated providers introduce new keys.
        """
        extra = set(extra_keys)
        instantiated: set[Key] = set()
        while True:
            known = set(self.underlying_graph.nodes) | extra
            providers: dict[Key, Provider] = {}
            for template in self._templates:
                for bound in forward_bindings(template, known):
                    provider = template.bind_type_vars(bound)
                    providers[provider.deduce_key()] = provider
            inserted = False
            for key, provider in providers.items():
                if key not in instantiated and not self._satisfied(key):
                    self.insert(provider)
                    instantiated.add(key)
                    inserted = True
            if not inserted:
                return

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
        if typevars := find_all_typevars(key):
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
        graph = self
        if self._templates:
            graph = self.copy()
            graph._instantiate_backward((key,))
        return graph._from_cyclebane(graph._cbgraph[key])

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
        graph = self
        if self._templates:
            # Mapping duplicates dependents of the mapped nodes, so generic
            # providers must be instantiated first.
            graph = self.copy()
            graph._instantiate_forward(node_values.keys())
        return graph._from_cyclebane(graph._cbgraph.map(node_values))

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
    if data_graph._templates:
        data_graph = data_graph.copy()
        data_graph._instantiate_backward(targets)
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
            spec = provider.arg_spec.map_keys(new_key.get, map_return=False)
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
