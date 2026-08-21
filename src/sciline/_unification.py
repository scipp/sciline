# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unification of generic type patterns with concrete keys.

Generic providers are not expanded eagerly. They are kept as templates and
instantiated on demand by unifying their argument and return-type patterns
with the concrete keys that appear in the pipeline (parameters, requested
targets, mapped keys). Constraints declared on type variables restrict which
concrete types they unify with.
"""

from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable
from types import UnionType
from typing import TYPE_CHECKING, Any, TypeVar, get_args, get_origin

if TYPE_CHECKING:
    from ._provider import Provider
    from .typing import Key


def find_all_typevars(t: type | TypeVar) -> set[TypeVar]:
    """Returns the set of all TypeVars in a type expression."""
    if isinstance(t, TypeVar):
        return {t}
    if params := getattr(t, '__parameters__', ()):
        return set(params)
    return set(itertools.chain(*map(find_all_typevars, get_args(t))))


def origin_and_args(t: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return the generic origin and args of ``t``, or ``(None, ())``.

    Supports regular typing generics as well as pydantic generic models, whose
    metaclass hides type parameters from :py:func:`typing.get_origin`.
    """
    if (origin := get_origin(t)) is not None:
        return origin, get_args(t)
    if (meta := getattr(t, '__pydantic_generic_metadata__', None)) is not None:
        if meta['origin'] is not None:
            return meta['origin'], meta['args']
    return None, ()


def parameterize(key: Key) -> Key:
    """Subscript bare generic classes with their own type parameters, recursively.

    E.g., for ``class A(Generic[T])``, turns ``A`` into ``A[T]`` and
    ``list[A]`` into ``list[A[T]]``, so that patterns have a uniform
    subscripted shape for unification.
    """
    origin, args = origin_and_args(key)
    if origin is not None:
        if origin is UnionType:
            return key
        return origin[tuple(parameterize(arg) for arg in args)]  # type: ignore[no-any-return]
    if params := getattr(key, '__parameters__', ()):
        return key[params]  # type: ignore[index, no-any-return]
    return key


def _pattern_origin_and_args(pattern: Any) -> tuple[Any, tuple[Any, ...]]:
    """Like :py:func:`origin_and_args`, but treats an unparametrized generic
    class whose subscription does not produce an inspectable alias (e.g. a
    pydantic model) as its own origin with its type parameters as args."""
    origin, args = origin_and_args(pattern)
    if origin is None and (params := getattr(pattern, '__parameters__', ())):
        return pattern, params
    return origin, args


def unify(pattern: Key | TypeVar, concrete: Key, bound: dict[TypeVar, Key]) -> bool:
    """Match ``concrete`` against ``pattern``, extending ``bound`` in place.

    Returns True on success. ``bound`` may contain partial bindings on failure
    and must be discarded by the caller in that case.
    """
    if isinstance(pattern, TypeVar):
        if pattern.__constraints__ and concrete not in pattern.__constraints__:
            return False
        if pattern in bound:
            return bound[pattern] == concrete
        bound[pattern] = concrete
        return True
    pattern_origin, pattern_args = _pattern_origin_and_args(pattern)
    if pattern_origin is None:
        return pattern == concrete
    concrete_origin, concrete_args = origin_and_args(concrete)
    if concrete_origin != pattern_origin:
        return False
    if len(pattern_args) != len(concrete_args):
        return False
    return all(
        unify(p, c, bound) for p, c in zip(pattern_args, concrete_args, strict=True)
    )


def subsumes(general: Key | TypeVar, specific: Key | TypeVar) -> bool:
    """Return whether every key matched by ``specific`` is matched by ``general``.

    One-sided unification: type variables of ``general`` may bind to
    sub-patterns of ``specific``, whose type variables are treated as opaque.
    Mutual subsumption means the patterns are equivalent up to renaming;
    one-sided subsumption means ``specific`` is strictly more specific.
    """
    return _subsumes(general, specific, {})


def _subsumes(
    general: Key | TypeVar, specific: Key | TypeVar, bound: dict[TypeVar, Any]
) -> bool:
    if isinstance(general, TypeVar):
        if general.__constraints__:
            if isinstance(specific, TypeVar):
                # ``specific`` matches keys in its own constraint set; all of
                # them must be admissible for ``general``.
                if not specific.__constraints__ or not set(
                    specific.__constraints__
                ) <= set(general.__constraints__):
                    return False
            elif specific not in general.__constraints__:
                return False
        if general in bound:
            return bool(bound[general] == specific)
        bound[general] = specific
        return True
    general_origin, general_args = _pattern_origin_and_args(general)
    if general_origin is None:
        return general == specific
    specific_origin, specific_args = _pattern_origin_and_args(specific)
    if specific_origin != general_origin:
        return False
    if len(general_args) != len(specific_args):
        return False
    return all(
        _subsumes(g, s, bound) for g, s in zip(general_args, specific_args, strict=True)
    )


def match_return(template: Provider, key: Key) -> Provider | None:
    """Instantiate a generic provider if its return type unifies with ``key``."""
    bound: dict[TypeVar, Key] = {}
    if unify(template.deduce_key(), key, bound):
        return template.bind_type_vars(bound)
    return None


def forward_bindings(
    template: Provider, known_keys: Iterable[Key]
) -> Generator[tuple[dict[TypeVar, Key], frozenset[Key]], None, None]:
    """Enumerate complete bindings of a template's TypeVars from known keys.

    Each generic argument of the template is unified with each known key;
    consistent combinations of the resulting bindings that bind all type
    variables of the template are yielded, together with the set of known keys
    that produced them. Arguments that match no known key are left unmatched,
    i.e., a binding is complete as long as the *other* arguments determine all
    type variables.
    """
    typevars = find_all_typevars(template.deduce_key())
    patterns = [p for p in template.arg_spec.keys() if find_all_typevars(p)]
    options = []
    for pattern in patterns:
        matches: list[tuple[dict[TypeVar, Key], Key | None]] = [({}, None)]
        for key in known_keys:
            bound: dict[TypeVar, Key] = {}
            if unify(pattern, key, bound):
                matches.append((bound, key))
        options.append(matches)
    seen = set()
    for combo in itertools.product(*options):
        merged: dict[TypeVar, Key] = {}
        if not all(_merge(merged, bound) for bound, _ in combo):
            continue
        if set(merged) != typevars:
            continue
        used = frozenset(key for _, key in combo if key is not None)
        if (fingerprint := (frozenset(merged.items()), used)) not in seen:
            seen.add(fingerprint)
            yield merged, used


def _merge(target: dict[TypeVar, Key], bound: dict[TypeVar, Key]) -> bool:
    for tv, key in bound.items():
        if target.setdefault(tv, key) != key:
            return False
    return True
