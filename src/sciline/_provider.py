# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
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

    def __init__(
        self, arg_names: tuple[str, ...], args: tuple[Key, ...], kwargs: dict[str, Key]
    ) -> None:
        """Build from components, use ``from_provider`` instead."""
        # Separate arg_names and args so args can be returned to scheduler as is
        self._arg_names = arg_names
        self._args = args
        self._kwargs = kwargs

    @classmethod
    def from_provider(cls, provider: Provider, bound: dict[TypeVar, Key]) -> ArgSpec:
        """Parse the argument spec of a provider."""
        hints = get_type_hints(provider)
        signature = inspect.getfullargspec(provider)
        arg_names = tuple(signature.args)
        args = tuple(
            _bind_free_typevars(hints[key], bound=bound) for key in signature.args
        )
        kwargs = {
            key: _bind_free_typevars(hints[key], bound=bound)
            for key in signature.kwonlyargs
        }
        return cls(arg_names, args, kwargs)

    @property
    def args(self) -> tuple[Key, ...]:
        return self._args

    @property
    def kwargs(self) -> dict[str, Key]:
        return self._kwargs

    def keys(self) -> Generator[Key, None, None]:
        """Flat iterator over all argument types."""
        yield from self._args
        yield from self._kwargs.values()

    def call(self, provider: Provider, values: dict[Key, Any]) -> Any:
        """Call a compatible provider with arguments extracted from ``values``."""
        return provider(
            *(values[arg] for arg in self._args),
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
