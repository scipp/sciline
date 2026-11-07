# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""A tree of pipelines for batch processing."""

from __future__ import annotations

import html
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx

from ._provider import Provider
from .handler import HandleAsComputeTimeException
from .pipeline import Pipeline, _is_multiple_keys

if TYPE_CHECKING:
    import graphviz

_LEVEL_COLORS = ["#a6cee3", "#fdbf6f", "#b2df8a", "#cab2d6", "#fb9a99"]


class MultiParam:
    """
    Marks a parameter value as a set of alternatives to be expanded into a new level
    of a :py:class:`BatchProcessor`.

    Parameters
    ----------
    values:
        Either a sequence of values (entries are labeled by ``str(value)``), or a
        mapping from label to value.
    """

    def __init__(self, values: Sequence[Any] | Mapping[str, Any]):
        if isinstance(values, Mapping):
            self.items = {str(k): v for k, v in values.items()}
        else:
            self.items = {str(v): v for v in values}
            if len(self.items) != len(values):
                raise ValueError(f"MultiParam labels are not unique: {list(values)}")

    def __repr__(self) -> str:
        return f"MultiParam({list(self.items)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MultiParam) and self.items == other.items

    __hash__ = None  # type: ignore[assignment]


def _is_label_mapping(spec: Mapping[Any, Any]) -> bool:
    # Pipeline keys are types, so a mapping with only str keys cannot be params.
    return len(spec) > 0 and all(isinstance(k, str) for k in spec)


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # e.g. ambiguous truth value of arrays
        return False


def _leaf_params(spec: Mapping[Any, Any]) -> Iterator[Mapping[Any, Any]]:
    if _is_label_mapping(spec):
        for sub in spec.values():
            yield from _leaf_params(sub)
    else:
        yield spec


def _varying_keys(specs: Iterable[Mapping[Any, Any]]) -> set[Any]:
    """Keys that are not set to the same value in all leaf parameter sets."""
    leaves = [leaf for spec in specs for leaf in _leaf_params(spec)]
    missing = object()
    keys = {k for leaf in leaves for k in leaf}
    return {
        k
        for k in keys
        if not all(
            _same(leaf.get(k, missing), leaves[0].get(k, missing)) for leaf in leaves
        )
    }


def _build(workflow: Pipeline, spec: Mapping[Any, Any]) -> Pipeline | BatchProcessor:
    """Build a node from a mapping of labels to specs or from a mapping of params."""
    if _is_label_mapping(spec):
        return BatchProcessor(
            {label: _build(workflow, sub) for label, sub in spec.items()},
            level_keys=_varying_keys(spec.values()),
        )
    multi = next(((k, v) for k, v in spec.items() if isinstance(v, MultiParam)), None)
    if multi is None:
        pipeline = workflow.copy()
        for key, value in spec.items():
            pipeline[key] = value
        return pipeline
    key, mp = multi
    # Replacing in place keeps the order of the remaining MultiParams.
    return BatchProcessor(
        {
            label: _build(workflow, {**spec, key: val})
            for label, val in mp.items.items()
        },
        level_keys={key},
        squeeze=True,
    )


def batch_processor(
    workflow: Pipeline, params: Mapping[str, Mapping[Any, Any]]
) -> BatchProcessor:
    """
    Create a :py:class:`BatchProcessor` by copying ``workflow`` for every leaf.

    Parameters
    ----------
    workflow:
        The pipeline to copy for every leaf.
    params:
        Either a dict of parameters containing at least one :py:class:`MultiParam`
        (see below), or a mapping from label to either

        - a dict of parameters (keys are not ``str``), optionally containing
          :py:class:`MultiParam` values which add a nested level (in dict order), or
        - a dict whose keys are all ``str``, interpreted as a further level of labels
          mapping to specs of the same form.
    """
    node = _build(workflow, params)
    if not isinstance(node, BatchProcessor):
        raise ValueError(
            "params must contain str labels or at least one MultiParam, "
            "otherwise there is nothing to batch over."
        )
    return node


class BatchProcessor:
    """
    A tree of :py:class:`Pipeline` objects, for batch processing.

    Use :py:func:`batch_processor` to build one from a workflow and
    parameters, or pass your own pipelines.

    Parameters
    ----------
    pipelines:
        Mapping from label to a pipeline, a collection, or a (nested) mapping of
        these. Labels must be strings.
    level_keys:
        The parameter keys that differ between the entries of this level. They are
        used to decide, for a given key, whether a value is specific to each entry
        (see :py:meth:`__setitem__`) and whether the level is omitted in
        :py:meth:`compute` (see ``squeeze``).
        If None (unknown), values are always set on all entries and the level is
        always kept in :py:meth:`compute`.
    squeeze:
        If True, :py:meth:`compute` computes targets that do not depend on any of
        ``level_keys`` only once and omits this level for them. Requires
        ``level_keys``.
    name:
        Name of this level, used by :py:meth:`visualize`. Defaults to the names of
        ``level_keys``.
    """

    def __init__(
        self,
        pipelines: Mapping[str, Pipeline | BatchProcessor | Mapping[str, Any]],
        *,
        level_keys: Iterable[Any] | None = None,
        squeeze: bool = False,
        name: str | None = None,
    ) -> None:
        if len(pipelines) == 0 or not all(isinstance(k, str) for k in pipelines):
            raise ValueError("pipelines must be a non-empty mapping with str keys.")
        if squeeze and level_keys is None:
            raise ValueError("squeeze=True requires level_keys.")
        self._level_keys = None if level_keys is None else frozenset(level_keys)
        self._squeeze = squeeze
        self._name = name
        self._children: dict[str, Pipeline | BatchProcessor] = {
            k: BatchProcessor(v) if isinstance(v, Mapping) else v
            for k, v in pipelines.items()
        }

    # -- container interface -------------------------------------------------

    def __getitem__(self, label: str) -> Pipeline | BatchProcessor:
        """Return the sub-collection or pipeline for ``label``."""
        return self._children[label]

    def __iter__(self) -> Iterator[str]:
        return iter(self._children)

    def __len__(self) -> int:
        return len(self._children)

    def keys(self) -> Iterable[str]:
        return self._children.keys()

    def items(self) -> Iterable[tuple[str, Pipeline | BatchProcessor]]:
        return self._children.items()

    def leaves(self) -> Iterator[Pipeline]:
        """Iterate over all pipelines in the tree."""
        for child in self._children.values():
            if isinstance(child, BatchProcessor):
                yield from child.leaves()
            else:
                yield child

    def __repr__(self) -> str:
        return f"BatchProcessor(keys={list(self._children)})"

    # -- modifying -----------------------------------------------------------

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Set ``key`` to ``value`` on the pipelines in the collection.

        If ``key`` depends on the parameters that vary over a level (as given by
        ``level_keys``), ``value`` must be a mapping (e.g. a
        :py:class:`scipp.DataGroup`) with exactly the labels of that level as keys;
        each entry is passed on to the corresponding child. This is the structure
        returned by :py:meth:`compute`, so ``bp[key] = bp.compute(key)`` works.
        Otherwise the same value is set on all children, whatever its type.
        """
        if self._depends_on_level(key, default=False):
            if not isinstance(value, Mapping) or set(value) != set(self._children):
                raise ValueError(
                    f"{key!r} depends on the parameters that vary over this level "
                    f"({sorted(map(str, self._level_keys))}), so the value must be "
                    f"a mapping with keys {list(self._children)}."
                )
            for label, child in self._children.items():
                child[key] = value[label]
            # The cached node no longer has the level keys as ancestors, so it must
            # be tracked as varying itself.
            self._level_keys = self._level_keys | {key}
        else:
            for child in self._children.values():
                child[key] = value

    def insert(self, provider: Provider, /) -> None:
        """Insert a provider in every pipeline in the collection."""
        for pipeline in self.leaves():
            pipeline.insert(provider)

    # -- computing -----------------------------------------------------------

    def _representative(self) -> Pipeline:
        child = next(iter(self._children.values()))
        return child if isinstance(child, Pipeline) else child._representative()

    def _depends_on_level(self, target: Any, *, default: bool) -> bool:
        """Whether target is, or is computed from, a key varying over this level."""
        if self._level_keys is None:
            return default
        graph = self._representative().underlying_graph
        if target not in graph:
            return True
        return target in self._level_keys or not self._level_keys.isdisjoint(
            nx.ancestors(graph, target)
        )

    def _level_name(self) -> str:
        if self._name is not None:
            return self._name
        if not self._level_keys:
            return "level"
        return ", ".join(
            sorted(getattr(k, "__name__", str(k)) for k in self._level_keys)
        )

    def _level_path(self) -> list[BatchProcessor]:
        """Levels from this one down to the leaves, following first children."""
        child = next(iter(self._children.values()))
        return [self] + (
            child._level_path() if isinstance(child, BatchProcessor) else []
        )

    def visualize(
        self,
        tp: Any = None,
        *,
        compact: bool = False,
        mode: Literal["data", "task", "both"] = "data",
        **kwargs: Any,
    ) -> graphviz.Digraph:
        """
        Return a graphviz graph of the workflow, annotated with the batch levels.

        The graph of a single representative pipeline is drawn. Nodes that differ
        between the entries of one or more levels are drawn as 3D boxes filled with
        the color of the level (striped boxes if they differ over several levels). A legend lists the
        levels with their number of entries. Values are never shown.

        Parameters
        ----------
        tp:
            Type or iterable of types to visualize. Defaults to the outputs of the
            pipeline.
        compact, mode, kwargs:
            Passed to :py:meth:`sciline.TaskGraph.visualize`.
        """
        rep = self._representative()
        if tp is None:
            tp = rep.output_keys()
        graph = rep.get(tp, handler=HandleAsComputeTimeException())
        levels = [lvl for lvl in self._level_path() if lvl._level_keys is not None]
        colors = [_LEVEL_COLORS[i % len(_LEVEL_COLORS)] for i in range(len(levels))]
        node_attrs: dict[Any, dict[str, str]] = {}
        for key in graph.keys():
            used = [
                colors[i]
                for i, lvl in enumerate(levels)
                if lvl._depends_on_level(key, default=False)
            ]
            if len(used) == 1:
                node_attrs[key] = {
                    "shape": "box3d",
                    "style": "filled",
                    "fillcolor": used[0],
                }
            elif used:
                # Gradients support only two colors, stripes any number.
                node_attrs[key] = {
                    "shape": "rectangle",
                    "style": "striped",
                    "fillcolor": ":".join(used),
                }
        dot = graph.visualize(
            compact=compact, mode=mode, node_attrs=node_attrs, **kwargs
        )
        if levels:
            rows = "".join(
                f'<tr><td bgcolor="{c}"> </td>'
                f"<td align=\"left\">{html.escape(lvl._level_name())} "
                f"&times;{len(lvl._children)}</td></tr>"
                for lvl, c in zip(levels, colors, strict=True)
            )
            dot.node(
                "_batch_legend",
                f'<<table border="0" cellborder="1" cellspacing="0">{rows}</table>>',
                shape="none",
            )
        return dot

    def _compute(self, targets: tuple[Any, ...], **kwargs: Any) -> dict[Any, Any]:
        import scipp as sc

        dependent = tuple(
            t
            for t in targets
            if not self._squeeze or self._depends_on_level(t, default=True)
        )
        independent = tuple(t for t in targets if t not in dependent)
        results: dict[Any, Any] = {}
        if independent:
            first = next(iter(self._children.values()))
            results.update(_compute_child(first, independent, **kwargs))
        if dependent:
            per_child = {
                label: _compute_child(child, dependent, **kwargs)
                for label, child in self._children.items()
            }
            for t in dependent:
                results[t] = sc.DataGroup({k: v[t] for k, v in per_child.items()})
        return results

    def compute(self, tp: Any, **kwargs: Any) -> Any:
        """
        Compute the result for the given key(s) on all pipelines.

        Returns a :py:class:`scipp.DataGroup` with one entry per label of the top
        level, each entry being a DataGroup for the next level, and so on. Levels
        created by a :py:class:`MultiParam` are omitted for targets that do not
        depend on the parameter (the result is computed once).

        Parameters
        ----------
        tp:
            Type, or iterable of types, to compute. For an iterable a dict mapping
            each type to its nested result is returned.
        kwargs:
            Passed on to :py:meth:`Pipeline.compute` (e.g. ``reporter``, ``scheduler``).
        """
        if _is_multiple_keys(tp):
            return self._compute(tuple(tp), **kwargs)
        return self._compute((tp,), **kwargs)[tp]


def _compute_child(
    child: Pipeline | BatchProcessor, targets: tuple[Any, ...], **kwargs: Any
) -> dict[Any, Any]:
    if isinstance(child, BatchProcessor):
        return child._compute(targets, **kwargs)
    return child.compute(targets, **kwargs)
