# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: PYI019
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
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
from ._unification import (
    find_all_typevars,
    key_depth,
    pattern_origin_and_args,
    match_return,
    parameterize,
    subsumes,
    unify,
)
from ._utils import key_name
from .handler import AmbiguousProvider, ErrorHandler, HandleAsBuildTimeException
from .typing import Graph, Key

if TYPE_CHECKING:
    import graphviz


def _as_graph(key: Key, value: Any) -> cb.Graph:
    """Create a cyclebane.Graph with a single value."""
    graph = nx.DiGraph()
    graph.add_node(key, value=value)
    return cb.Graph(graph)


T = TypeVar('T', bound='DataGraph')

_providing_attrs = frozenset(('value', 'provider', 'reduce'))

_no_value = object()

_max_key_depth = 32


@dataclass
class _TemplateValue:
    """A value set for a generic key, applied to demanded specializations."""

    pattern: Key
    value: Any


def _template_pattern(template: Provider | _TemplateValue) -> Key:
    if isinstance(template, _TemplateValue):
        return template.pattern
    return template.deduce_key()  # type: ignore[no-any-return]


def _equivalent(a: Key, b: Key) -> bool:
    """Whether two patterns match the same keys (equal up to renaming)."""
    return subsumes(a, b) and subsumes(b, a)


class DataGraph:
    def __init__(self, providers: None | Iterable[ToProvider | Provider]) -> None:
        self._templates: list[Provider | _TemplateValue] = []
        self._cbgraph = cb.Graph(nx.DiGraph())
        for provider in providers or []:
            self.insert(provider)

    def _from_cyclebane(self: T, graph: cb.Graph) -> T:
        out = type(self)([])
        out._cbgraph = graph
        out._templates = list(self._templates)
        return out

    @property
    def _has_templates(self) -> bool:
        return bool(self._templates)

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
        if find_all_typevars(return_type):
            self._register_template(provider)
            return
        # Trigger UnboundTypeVar error if any input typevars are not bound
        provider = provider.bind_type_vars({})
        self._get_clean_node(return_type)['provider'] = provider
        for dep in provider.arg_spec.keys():
            self.underlying_graph.add_edge(dep, return_type, key=dep)

    def _register_template(self, provider: Provider) -> None:
        """Store a generic provider for on-demand instantiation.

        Generic providers are instantiated by unifying their type patterns
        with the concrete keys demanded from the graph, see
        :py:meth:`_instantiate_backward`.
        """
        spec = provider.arg_spec.map_keys(parameterize)
        provider = Provider(func=provider.func, arg_spec=spec, kind=provider.kind)
        return_type = provider.deduce_key()
        if isinstance(return_type, TypeVar) and not return_type.__constraints__:
            raise ValueError(
                f"Provider {provider} returns a bare unconstrained type variable "
                f"{return_type!r}, which would match any requested key. Use a "
                "generic class as return type or constrain the type variable."
            )
        typevars = find_all_typevars(return_type)
        arg_typevars: set[TypeVar] = set()
        for arg in provider.arg_spec.keys():
            arg_typevars |= find_all_typevars(arg)
        if unbound := arg_typevars - typevars:
            raise UnboundTypeVar(
                f"Provider {provider} has type variables {unbound} in its "
                "arguments that do not appear in its return type."
            )
        self._add_template(provider)

    def _register_template_value(self, key: Key, value: Any) -> None:
        """Store a value for a generic key, applied to all demanded specializations."""
        pattern = parameterize(key)
        self._add_template(_TemplateValue(pattern=pattern, value=value))

    def _add_template(self, template: Provider | _TemplateValue) -> None:
        # A template with an equivalent pattern is replaced, mirroring the
        # replacement semantics of concrete keys.
        pattern = _template_pattern(template)
        self._templates = [
            t for t in self._templates if not _equivalent(_template_pattern(t), pattern)
        ]
        self._templates.append(template)

    def _matching_template(self, key: Key) -> Provider | _TemplateValue | None:
        """Return the template that provides ``key``, or None.

        Resolution is order-independent: registration already replaced
        templates with equivalent patterns, and among the remaining matches the
        strictly most specific pattern wins. Incomparable overlapping matches
        raise :py:class:`AmbiguousProvider`. A returned provider is already
        bound to ``key``.
        """
        matches: list[tuple[Key, Provider | _TemplateValue]] = []
        for template in self._templates:
            if isinstance(template, _TemplateValue):
                bound: dict[TypeVar, Key] = {}
                if unify(template.pattern, key, bound):
                    matches.append((template.pattern, template))
            elif (provider := match_return(template, key)) is not None:
                matches.append((template.deduce_key(), provider))
        if len(matches) < 2:
            return matches[0][1] if matches else None

        def is_dominated(index: int) -> bool:
            pattern = matches[index][0]
            return any(
                i != index and subsumes(pattern, other) and not subsumes(other, pattern)
                for i, (other, _) in enumerate(matches)
            )

        remaining = [m for i, m in enumerate(matches) if not is_dominated(i)]
        if len(remaining) == 1:
            return remaining[0][1]
        names = ', '.join(
            resolved.location.qualname
            if isinstance(resolved, Provider)
            else f"value for '{key_name(pattern)}'"
            for pattern, resolved in remaining
        )
        raise AmbiguousProvider(
            f"Multiple incomparable generic providers match '{key_name(key)}': "
            f"{names}. Insert a provider with this exact return pattern, or a "
            "more specific one, to disambiguate."
        )

    def _satisfied(self, key: Key) -> bool:
        graph = self.underlying_graph
        if key in graph and bool(graph.nodes[key].keys() & _providing_attrs):
            return True
        # Mapped roots receive their values from the mapping.
        return key in self._cbgraph.value_keys

    def _consumed_by_template(self, key: Key) -> bool:
        """Whether any generic provider has an argument matching ``key``."""
        for template in self._templates:
            if not isinstance(template, Provider):
                continue
            for arg in template.arg_spec.keys():
                bound: dict[TypeVar, Key] = {}
                if unify(arg, key, bound):
                    return True
        return False

    def _demanded(self: T, targets: Iterable[Key]) -> T:
        """Return self, or a copy with templates instantiated for the targets."""
        if not self._has_templates:
            return self
        out = self.copy()
        out._instantiate_backward(targets)
        return out

    def _instantiate_backward(self, keys: Iterable[Key]) -> None:
        """Instantiate templates for demanded keys and their dependencies."""
        stack = list(keys)
        seen = set()
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            if key_depth(key) > _max_key_depth:
                raise RuntimeError(
                    f"Nesting depth of demanded key '{key_name(key)}' exceeds "
                    f"{_max_key_depth}; this usually indicates a non-terminating "
                    "chain of generic providers."
                )
            if not self._satisfied(key):
                template = self._matching_template(key)
                if isinstance(template, _TemplateValue):
                    self[key] = template.value
                elif template is not None:
                    self.insert(template)
            if key in self.underlying_graph:
                stack.extend(self.underlying_graph.predecessors(key))

    def _template_output_patterns(self) -> list[Key]:
        """Return-type patterns of generic providers not consumed by templates.

        Whether one generic provider consumes another's output cannot be
        decided exactly without instantiation; consumption is approximated by
        comparing pattern origins.
        """
        providers = [t for t in self._templates if isinstance(t, Provider)]
        consumed = {
            origin
            for template in providers
            for arg in template.arg_spec.keys()
            if find_all_typevars(arg)
            and (origin := pattern_origin_and_args(arg)[0]) is not None
        }
        return [
            pattern
            for template in providers
            if pattern_origin_and_args(pattern := template.deduce_key())[0]
            not in consumed
        ]

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
        if find_all_typevars(key):
            self._register_template_value(key, value)
            return

        # TODO If key is generic, should we support multi-sink case and update all?
        # Would imply that we need the same for __getitem__.
        self._cbgraph[key] = (
            value._cbgraph if isinstance(value, DataGraph) else _as_graph(key, value)
        )

    def __getitem__(self: T, key: Key) -> T:
        """Return the subgraph that computes the given key.

        All generic providers are retained in the subgraph; they are only
        instantiated on demand.
        """
        graph = self._demanded((key,))
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
        # Note that dependents of the mapped nodes need not exist yet: which
        # nodes carry which indices is derived at task-graph build time, so
        # providers instantiated on demand after mapping are handled correctly.
        return self._from_cyclebane(self._cbgraph.map(node_values))

    def reduce(
        self: T, *, func: Callable[..., Any], key: Key | None = None, **kwargs: Any
    ) -> T:
        """Reduce the outputs of a mapped graph into a single value and provider.

        Parameters
        ----------
        func:
            Function that takes the values to reduce and returns a single value.
            This function is passed as many arguments as there are values to reduce.
        key:
            The node to reduce. If not given, the unique sink of the graph is
            used; this requires that the sink cannot come from a generic
            provider that has not been instantiated yet.
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
        graph = self
        if self._has_templates:
            if key is None:
                sinks = [
                    node
                    for node, degree in self.underlying_graph.out_degree
                    if degree == 0
                ]
                if len(sinks) != 1 or self._consumed_by_template(sinks[0]):
                    raise ValueError(
                        "Cannot determine which node to reduce: the pipeline "
                        "contains generic providers, so the reduced node may "
                        "not have been instantiated yet. Pass an explicit "
                        "'key'."
                    )
            else:
                # The reduced key is a demand; instantiate providers for it.
                graph = self._demanded((key,))
        return graph._from_cyclebane(
            graph._cbgraph.reduce(key=key, attrs={'reduce': func}, **kwargs)
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


def to_task_graph(
    data_graph: DataGraph, targets: tuple[Key, ...], handler: ErrorHandler | None = None
) -> Graph:
    data_graph = data_graph._demanded(targets)
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
