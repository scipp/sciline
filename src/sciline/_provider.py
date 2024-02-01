# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Sequence,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    from .typing import Key, Provider


class UnboundTypeVar(Exception):
    """
    Raised when a parameter of a generic provider is not bound to a concrete type.
    """


class ArgSpec:
    """Argument specification for a provider."""

    def __init__(self, *, args: dict[str, Key], kwargs: dict[str, Key]) -> None:
        """Build from components, use dedicated creation functions instead."""
        self._args = args
        self._kwargs = kwargs

    @classmethod
    def from_provider(cls, provider: Provider, bound: dict[TypeVar, Key]) -> ArgSpec:
        """Parse the argument spec of a provider."""
        hints = get_type_hints(provider)
        signature = inspect.getfullargspec(provider)
        args = {
            name: _bind_free_typevars(hints[name], bound=bound)
            for name in signature.args
        }
        kwargs = {
            name: _bind_free_typevars(hints[name], bound=bound)
            for name in signature.kwonlyargs
        }
        return cls(args=args, kwargs=kwargs)

    @classmethod
    def from_arg(cls, arg: Key) -> ArgSpec:
        """Create ArgSpec from a single positional argument."""
        return cls(args={'unknown': arg}, kwargs={})

    @classmethod
    def null(cls) -> ArgSpec:
        """Create ArgSpec for a nullary function (no args)."""
        return cls(args={}, kwargs={})

    @property
    def args(self) -> Generator[Key, None, None]:
        yield from self._args.values()

    @property
    def kwargs(self) -> Generator[tuple[str, Key], None, None]:
        yield from self._kwargs.items()

    def keys(self) -> Generator[Key, None, None]:
        """Flat iterator over all argument types."""
        yield from self._args.values()
        yield from self._kwargs.values()

    def filter_in_keys(self, select: Sequence[Key]) -> ArgSpec:
        """Return a new ArgSpec that only has keys that are present in ``select``."""

        def do_filter(d: dict[str, Key]) -> dict[str, Key]:
            return {name: arg for name, arg in d.items() if arg in select}

        return ArgSpec(
            args=do_filter(self._args),
            kwargs=do_filter(self._kwargs),
        )

    def call(self, provider: Provider, values: dict[Key, Any]) -> Any:
        """Call a compatible provider with arguments extracted from ``values``."""
        return provider(
            *(values[arg] for arg in self._args.values()),
            **{key: values[arg] for key, arg in self._kwargs.items()},
        )


def _bind_free_typevars(tp: TypeVar | Key, bound: dict[TypeVar, Key]) -> Key:
    if isinstance(tp, TypeVar):
        if (result := bound.get(tp)) is None:
            raise UnboundTypeVar(f'Unbound type variable {tp}')
        return result
    elif (origin := get_origin(tp)) is not None:
        result = origin[tuple(_bind_free_typevars(arg, bound) for arg in get_args(tp))]
        if result is None:
            raise ValueError(f'Binding type variables in {tp} resulted in `None`')
        return result
    else:
        return tp
