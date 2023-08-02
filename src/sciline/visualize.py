# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Callable, Dict, List, Tuple, Union, get_args, get_origin

from graphviz import Digraph

from .pipeline import Item, Label
from .scheduler import Graph


def to_graphviz(graph: Graph, **kwargs: Any) -> Digraph:
    """
    Convert output of :py:class:`sciline.Pipeline.get_graph` to a graphviz graph.

    Parameters
    ----------
    graph:
        Output of :py:class:`sciline.Pipeline.get_graph`.
    kwargs:
        Keyword arguments passed to :py:class:`graphviz.Digraph`.
    """
    dot = Digraph(strict=True, **kwargs)
    for p, (p_name, args, ret) in _format_graph(graph).items():
        if '(' in ret:
            dot.node(ret, ret, shape='box3d')
        else:
            dot.node(ret, ret, shape='rectangle')
        # Do not draw dummy providers created by Pipeline when setting instances
        if p_name in (
            'Pipeline.__setitem__.<locals>.<lambda>',
            'Pipeline.set_param_table.<locals>.<lambda>',
        ):
            continue
        # Do not draw the internal provider gathering index-dependent results into
        # a dict
        if p_name.startswith('Pipeline._build_indexed_subgraph.'):
            for arg in args:
                dot.edge(arg, ret, style='dashed')
        else:
            dot.node(p, p_name, shape='ellipse')
            for arg in args:
                dot.edge(arg, p)
            dot.edge(p, ret)
    return dot


def _format_graph(graph: Graph) -> Dict[str, Tuple[str, List[str], str]]:
    return {
        _format_provider(provider, ret): (
            provider.__qualname__,
            [_format_type(a) for a in args],
            _format_type(ret),
        )
        for ret, (provider, args) in graph.items()
    }


def _format_provider(provider: Callable[..., Any], ret: type) -> str:
    return f'{provider.__qualname__}_{_format_type(ret)}'


def _extract_type_and_labels(key: Union[Item, Label, type]) -> Tuple[type, List[type]]:
    if isinstance(key, Item):
        label = key.label
        return key.tp, [lb.tp for lb in label]
    if isinstance(key, Label):
        tp, labels = _extract_type_and_labels(key.tp)
        return tp, [key.tp] + labels
    return key, []


def _format_type(tp: type) -> str:
    """
    Helper for _format_graph.

    If tp is a generic such as Array[float], we want to return 'Array[float]',
    but strip all module prefixes from the type name as well as the params.
    We may make this configurable in the future.
    """

    tp, labels = _extract_type_and_labels(tp)

    def get_base(tp: type) -> str:
        return tp.__name__ if hasattr(tp, '__name__') else str(tp).split('.')[-1]

    def with_labels(base: str) -> str:
        if labels:
            return f'{base}({", ".join([get_base(l) for l in labels])})'
        return base

    if (origin := get_origin(tp)) is not None:
        params = [_format_type(param) for param in get_args(tp)]
        return with_labels(f'{get_base(origin)}[{", ".join(params)}]')
    else:
        return with_labels(get_base(tp))
