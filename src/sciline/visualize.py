# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from graphviz import Digraph

from ._provider import Provider, ProviderKind
from .typing import Graph, Item, Key, get_optional


@dataclass
class Node:
    name: str
    collapsed: bool = False


@dataclass
class FormattedProvider:
    name: str
    args: list[Node]
    ret: Node
    kind: ProviderKind


FormattedGraph = Dict[str, FormattedProvider]


def to_graphviz(
    graph: Graph,
    compact: bool = False,
    cluster_generics: bool = True,
    cluster_color: Optional[str] = '#f0f0ff',
    **kwargs: Any,
) -> Digraph:
    """
    Convert output of :py:class:`sciline.Pipeline.get_graph` to a graphviz graph.

    Parameters
    ----------
    graph:
        Output of :py:class:`sciline.Pipeline.get_graph`.
    compact:
        If True, parameter-table-dependent branches are collapsed into a single copy
        of the branch. Recommended for large graphs with long parameter tables.
    cluster_generics:
        If True, generic products are grouped into clusters.
    cluster_color:
        Background color of clusters. If None, clusters are dotted.
    kwargs:
        Keyword arguments passed to :py:class:`graphviz.Digraph`.
    """
    dot = Digraph(strict=True, **kwargs)
    formatted_graph = _format_graph(graph, compact=compact)
    ordered_graph = dict(
        sorted(formatted_graph.items(), key=lambda item: item[1].ret.name)
    )
    subgraphs = _to_subgraphs(ordered_graph)

    for origin, subgraph in subgraphs.items():
        cluster = cluster_generics and len(subgraph) > 1
        name = f'cluster_{origin}' if cluster else None
        with dot.subgraph(name=name) as dot_subgraph:
            if cluster:
                dot_subgraph.attr(rank='same')
                if cluster_color is None:
                    dot_subgraph.attr(style='dotted')
                else:
                    dot_subgraph.attr(style='filled', color=cluster_color)
            _add_subgraph(subgraph, dot, dot_subgraph)
    return dot


def _to_subgraphs(graph: FormattedGraph) -> Dict[str, FormattedGraph]:
    def get_subgraph_name(name: str, kind: str) -> str:
        if kind == 'series':
            # Example: Series[RowId, Material[Country]] -> RowId, Material[Country]
            return name.partition('[')[-1].rpartition(']')[0]
        return name.split('[')[0]

    subgraphs: Dict[str, FormattedGraph] = {}
    for p, formatted_p in graph.items():
        subgraph_name = get_subgraph_name(formatted_p.ret.name, formatted_p.kind)
        subgraphs.setdefault(subgraph_name, {})
        subgraphs[subgraph_name][p] = formatted_p
    return subgraphs


def _add_subgraph(graph: FormattedGraph, dot: Digraph, subgraph: Digraph) -> None:
    for p, formatted_p in graph.items():
        if formatted_p.kind == 'unsatisfied':
            subgraph.node(
                formatted_p.ret.name,
                formatted_p.ret.name,
                shape='box3d' if formatted_p.ret.collapsed else 'rectangle',
                color='red',
                fontcolor='red',  # Set text color to red
                style='dashed',
            )
        else:
            subgraph.node(
                formatted_p.ret.name,
                formatted_p.ret.name,
                shape='box3d' if formatted_p.ret.collapsed else 'rectangle',
            )
        # Do not draw the internal provider gathering index-dependent results into
        # a dict
        if formatted_p.kind == 'series':
            for arg in formatted_p.args:
                dot.edge(arg.name, formatted_p.ret.name, style='dashed')
        elif formatted_p.kind == 'function':
            dot.node(p, formatted_p.name, shape='ellipse')
            for arg in formatted_p.args:
                dot.edge(arg.name, p)
            dot.edge(p, formatted_p.ret.name)
        # else: Do not draw dummy providers created by Pipeline when setting instances


def _qualname(obj: Any) -> Any:
    return (
        obj.__qualname__ if hasattr(obj, '__qualname__') else obj.__class__.__qualname__
    )


def _format_graph(graph: Graph, compact: bool) -> FormattedGraph:
    return {
        _format_provider(provider, ret, compact=compact): FormattedProvider(
            name=_qualname(provider.func),
            args=[_format_type(a, compact=compact) for a in provider.arg_spec.keys()],
            ret=_format_type(ret, compact=compact),
            kind=provider.kind,
        )
        for ret, provider in graph.items()
    }


def _format_provider(provider: Provider, ret: Key, compact: bool) -> str:
    return f'{provider.location.qualname}_{_format_type(ret, compact=compact).name}'


T = TypeVar('T')


def _extract_type_and_labels(
    key: Union[Item[T], Type[T]], compact: bool
) -> Tuple[Type[T], List[Union[type, Tuple[type, Any]]]]:
    if isinstance(key, Item):
        label = key.label
        return key.tp, [lb.tp if compact else (lb.tp, lb.index) for lb in label]
    return key, []


def _format_type(tp: Key, compact: bool = False) -> Node:
    """
    Helper for _format_graph.

    If tp is a generic such as Array[float], we want to return 'Array[float]',
    but strip all module prefixes from the type name as well as the params.
    We may make this configurable in the future.
    """

    tp, labels = _extract_type_and_labels(tp, compact=compact)

    if (tp_ := get_optional(tp)) is not None:
        tp = tp_

    def get_base(tp: Key) -> str:
        return tp.__name__ if hasattr(tp, '__name__') else str(tp).split('.')[-1]

    def format_label(label: Union[type, Tuple[type, Any]]) -> str:
        if isinstance(label, tuple):
            tp, index = label
            return f'{get_base(tp)}={index}'
        return get_base(label)

    def with_labels(base: str) -> Node:
        if labels:
            return Node(
                name=f'{base}({", ".join([format_label(l) for l in labels])})',
                collapsed=compact,
            )
        return Node(name=base)

    if (origin := get_origin(tp)) is not None:
        params = [_format_type(param).name for param in get_args(tp)]
        return with_labels(f'{get_base(origin)}[{", ".join(params)}]')
    else:
        return with_labels(get_base(tp))
