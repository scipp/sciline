# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Unification of generic type patterns with concrete keys.

Generic providers whose type variables lack constraints are not expanded
eagerly. They are kept as templates and instantiated on demand by unifying
their argument and return-type patterns with the concrete keys that appear in
the pipeline (parameters, requested targets, mapped keys).
"""

from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING, TypeVar, get_args, get_origin

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
    if (origin := get_origin(pattern)) is None:
        return pattern == concrete
    if get_origin(concrete) != origin:
        return False
    pattern_args = get_args(pattern)
    concrete_args = get_args(concrete)
    if len(pattern_args) != len(concrete_args):
        return False
    return all(
        unify(p, c, bound) for p, c in zip(pattern_args, concrete_args, strict=True)
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
