# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Literal,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

if TYPE_CHECKING:
    from .typing import Key


ToProvider = Callable[..., Any]
ProviderKind = Literal['function', 'parameter', 'series', 'table', 'sentinel']


class UnboundTypeVar(Exception):
    """
    Raised when a parameter of a generic provider is not bound to a concrete type.
    """


class Provider:
    def __init__(
        self,
        *,
        func: ToProvider,
        arg_spec: ArgSpec,
        kind: ProviderKind,
        location: Optional[ProviderLocation] = None,
    ) -> None:
        self._func = func
        self._arg_spec = arg_spec
        self._kind = kind
        self._location = (
            location if location is not None else ProviderLocation.from_function(func)
        )

    @classmethod
    def from_function(cls, func: ToProvider) -> Provider:
        return cls(func=func, arg_spec=ArgSpec.from_function(func), kind='function')

    @classmethod
    def parameter(cls, param: Any) -> Provider:
        return cls(
            func=lambda: param,
            arg_spec=ArgSpec.null(),
            kind='parameter',
            location=ProviderLocation(
                name=f'param({type(param).__name__})', module='sciline'
            ),
        )

    @classmethod
    def table(cls, param: Any) -> Provider:
        return cls(
            func=lambda: param,
            arg_spec=ArgSpec.null(),
            kind='table',
            location=ProviderLocation(
                name=f'table({type(param).__name__})', module='sciline'
            ),
        )

    @classmethod
    def provide_none(cls) -> Provider:
        """Provider that takes no arguments and returns None."""
        return cls(
            func=lambda: None,
            arg_spec=ArgSpec.null(),
            kind='function',
            location=ProviderLocation(name='provide_none', module='sciline'),
        )

    @property
    def func(self) -> ToProvider:
        return self._func

    @property
    def arg_spec(self) -> ArgSpec:
        return self._arg_spec

    @property
    def kind(self) -> ProviderKind:
        return self._kind

    @property
    def location(self) -> ProviderLocation:
        return self._location

    @property
    def qualname(self) -> str:
        # TODO merge with location
        return getattr(self._func, '__qualname__', self._func.__class__.__qualname__)

    def deduce_key(self) -> Any:
        if (key := get_type_hints(self._func).get('return')) is None:
            raise ValueError(
                f'Provider {self} lacks type-hint for return value or returns NOne.'
            )
        return key

    def bind(self, bound: dict[TypeVar, Key]) -> Provider:
        return Provider(
            func=self._func, arg_spec=self._arg_spec.bind(bound), kind=self._kind
        )

    def map_arg_keys(self, transform: Callable[[Key], Key]) -> Provider:
        return Provider(
            func=self._func,
            arg_spec=self._arg_spec.map_keys(transform),
            kind=self._kind,
        )

    def __str__(self) -> str:
        return f"Provider('{self.location.name}')"

    def __repr__(self) -> str:
        return (
            f"Provider('{self.location.module}.{self.location.name}', "
            f"func={self._func})"
        )

    def call(self, values: dict[Key, Any]) -> Any:
        """Call the provider with arguments extracted from ``values``."""
        return self._func(
            *(values[arg] for arg in self._arg_spec.args),
            **{key: values[arg] for key, arg in self._arg_spec.kwargs},
        )


class ArgSpec:
    """Argument specification for a provider."""

    def __init__(self, *, args: dict[str, Key], kwargs: dict[str, Key]) -> None:
        """Build from components, use dedicated creation functions instead."""
        self._args = args
        self._kwargs = kwargs

    @classmethod
    def from_function(cls, provider: ToProvider) -> ArgSpec:
        """Parse the argument spec of a provider."""
        hints = get_type_hints(provider)
        signature = inspect.getfullargspec(provider)
        args = {name: hints[name] for name in signature.args}
        kwargs = {name: hints[name] for name in signature.kwonlyargs}
        return cls(args=args, kwargs=kwargs)

    @classmethod
    def from_args(cls, *args: Key) -> ArgSpec:
        """Create ArgSpec from positional arguments."""
        return cls(args={f'unknown_{i}': arg for i, arg in enumerate(args)}, kwargs={})

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

    def bind(self, bound: dict[TypeVar, Key]) -> ArgSpec:
        """Bind concrete types to TypeVars."""
        return self.map_keys(lambda arg: _bind_free_typevars(arg, bound=bound))

    def map_keys(self, transform: Callable[[Key], Key]) -> ArgSpec:
        """Return a new ArgSpec with the keys mapped by ``callback``."""
        return ArgSpec(
            args={name: transform(arg) for name, arg in self._args.items()},
            kwargs={name: transform(arg) for name, arg in self._kwargs.items()},
        )


@dataclass
class ProviderLocation:
    name: str
    module: str

    @classmethod
    def from_function(cls, func: ToProvider) -> ProviderLocation:
        return cls(
            name=func.__name__, module=getattr(inspect.getmodule(func), '__name__', '')
        )


def _bind_free_typevars(tp: Union[TypeVar, Key], bound: dict[TypeVar, Key]) -> Key:
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