# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from graphviz import Digraph

from .pipeline import Pipeline, SeriesProvider
from .typing import Graph, Item, Key


def to_graphviz(graph: Graph, compact: bool = False, **kwargs: Any) -> Digraph:
    """
    Convert output of :py:class:`sciline.Pipeline.get_graph` to a graphviz graph.

    Parameters
    ----------
    graph:
        Output of :py:class:`sciline.Pipeline.get_graph`.
    compact:
        If True, parameter-table-dependent branches are collapsed into a single copy
        of the branch. Recommendend for large graphs with long parameter tables.
    kwargs:
        Keyword arguments passed to :py:class:`graphviz.Digraph`.
    """
    dot = Digraph(strict=True, **kwargs)
    for p, (p_name, args, ret) in _format_graph(graph, compact=compact).items():
        if '(' in ret and '=' not in ret:
            dot.node(ret, ret, shape='box3d')
        else:
            dot.node(ret, ret, shape='rectangle')
        # Do not draw dummy providers created by Pipeline when setting instances
        if p_name in (
            f'{_qualname(Pipeline.__setitem__)}.<locals>.<lambda>',
            f'{_qualname(Pipeline.set_param_table)}.<locals>.<lambda>',
        ):
            continue
        # Do not draw the internal provider gathering index-dependent results into
        # a dict
        if p_name == _qualname(SeriesProvider):
            for arg in args:
                dot.edge(arg, ret, style='dashed')
        else:
            dot.node(p, p_name, shape='ellipse')
            for arg in args:
                dot.edge(arg, p)
            dot.edge(p, ret)
    return dot


def _qualname(obj: Any) -> Any:
    return (
        obj.__qualname__ if hasattr(obj, '__qualname__') else obj.__class__.__qualname__
    )


def _format_graph(graph: Graph, compact: bool) -> Dict[str, Tuple[str, List[str], str]]:
    return {
        _format_provider(provider, ret, compact=compact): (
            _qualname(provider),
            [_format_type(a, compact=compact) for a in args],
            _format_type(ret, compact=compact),
        )
        for ret, (provider, args) in graph.items()
    }


def _format_provider(provider: Callable[..., Any], ret: Key, compact: bool) -> str:
    return f'{_qualname(provider)}_{_format_type(ret, compact=compact)}'


T = TypeVar('T')


def _extract_type_and_labels(
    key: Union[Item[T], Type[T]], compact: bool
) -> Tuple[Type[T], List[Union[type, Tuple[type, Any]]]]:
    if isinstance(key, Item):
        label = key.label
        return key.tp, [lb.tp if compact else (lb.tp, lb.index) for lb in label]
    return key, []


def _format_type(tp: Key, compact: bool = False) -> str:
    """
    Helper for _format_graph.

    If tp is a generic such as Array[float], we want to return 'Array[float]',
    but strip all module prefixes from the type name as well as the params.
    We may make this configurable in the future.
    """

    tp, labels = _extract_type_and_labels(tp, compact=compact)

    def get_base(tp: type) -> str:
        return tp.__name__ if hasattr(tp, '__name__') else str(tp).split('.')[-1]

    def format_label(label: Union[type, Tuple[type, Any]]) -> str:
        if isinstance(label, tuple):
            tp, index = label
            return f'{get_base(tp)}={index}'
        return get_base(label)

    def with_labels(base: str) -> str:
        if labels:
            return f'{base}({", ".join([format_label(l) for l in labels])})'
        return base

    if (origin := get_origin(tp)) is not None:
        params = [_format_type(param) for param in get_args(tp)]
        return with_labels(f'{get_base(origin)}[{", ".join(params)}]')
    else:
        return with_labels(get_base(tp))
