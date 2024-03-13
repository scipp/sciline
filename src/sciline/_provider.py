# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Handling of providers and their arguments."""
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
"""Callable that can be converted to a provider."""

ProviderKind = Literal[
    'function', 'parameter', 'series', 'table_cell', 'sentinel', 'unsatisfied'
]
"""Identifies the kind of a provider, most are used internally."""


class UnboundTypeVar(Exception):
    """
    Raised when a parameter of a generic provider is not bound to a concrete type.
    """


class Provider:
    """A provider.

    This class wraps a function that returns the provided values.
    That function can be a user-provided callable, in which case
    ``kind = 'function'``, or an internally constructed function
    for providing parameters or other special values.
    """

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
        """Construct from a function or other callable."""
        return cls(func=func, arg_spec=ArgSpec.from_function(func), kind='function')

    @classmethod
    def parameter(cls, param: Any) -> Provider:
        """Construct a provider that always returns the given value."""
        return cls(
            func=lambda: param,
            arg_spec=ArgSpec.null(),
            kind='parameter',
            location=ProviderLocation(
                name=f'param({type(param).__name__})', module=_module_name(param)
            ),
        )

    @classmethod
    def table_cell(cls, param: Any) -> Provider:
        """Construct a provider that returns the label for a table row."""
        return cls(
            func=lambda: param,
            arg_spec=ArgSpec.null(),
            kind='table_cell',
            location=ProviderLocation(
                name=f'table_cell({type(param).__name__})', module=_module_name(param)
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
        """Return the function that implements the provider."""
        return self._func

    @property
    def arg_spec(self) -> ArgSpec:
        """Return the argument specification for the provider."""
        return self._arg_spec

    @property
    def kind(self) -> ProviderKind:
        """Return the kind of the provider."""
        return self._kind

    @property
    def location(self) -> ProviderLocation:
        """Return the location of the provider in source code."""
        return self._location

    def deduce_key(self) -> Any:
        """Attempt to determine the key (return type) of the provider."""
        if (key := self._arg_spec.return_) is None:
            raise ValueError(
                f'Provider {self} lacks type-hint for return value or returns None.'
            )
        return key

    def bind_type_vars(self, bound: dict[TypeVar, Key]) -> Provider:
        """Replace TypeVars with their corresponding keys."""
        return Provider(
            func=self._func,
            arg_spec=self._arg_spec.bind_type_vars(bound),
            kind=self._kind,
        )

    def map_arg_keys(self, transform: Callable[[Key], Key]) -> Provider:
        """Return a new provider with transformed argument keys."""
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

    def __init__(
        self, *, args: dict[str, Key], kwargs: dict[str, Key], return_: Optional[Key]
    ) -> None:
        """Build from components, use dedicated creation functions instead."""
        self._args = args
        self._kwargs = kwargs
        self._return = return_

    @classmethod
    def from_function(cls, provider: ToProvider) -> ArgSpec:
        """Parse the argument spec of a provider."""
        args, kwonlyargs, hints = _get_args_and_types(provider)
        try:
            arg_hints = {name: hints[name] for name in args}
            kwarg_hints = {name: hints[name] for name in kwonlyargs}
        except KeyError:
            raise ValueError(
                f'Provider {provider} lacks type-hint for arguments.'
            ) from None
        return cls(args=arg_hints, kwargs=kwarg_hints, return_=hints.get('return'))

    @classmethod
    def from_args(cls, *args: Key) -> ArgSpec:
        """Create ArgSpec from positional arguments."""
        return cls(
            args={f'unknown_{i}': arg for i, arg in enumerate(args)},
            kwargs={},
            return_=None,
        )

    @classmethod
    def null(cls) -> ArgSpec:
        """Create ArgSpec for a nullary function (no args)."""
        return cls(args={}, kwargs={}, return_=None)

    @property
    def args(self) -> Generator[Key, None, None]:
        yield from self._args.values()

    @property
    def kwargs(self) -> Generator[tuple[str, Key], None, None]:
        yield from self._kwargs.items()

    @property
    def return_(self) -> Optional[Key]:
        return self._return

    def keys(self) -> Generator[Key, None, None]:
        """Flat iterator over all argument types."""
        yield from self._args.values()
        yield from self._kwargs.values()

    def bind_type_vars(self, bound: dict[TypeVar, Key]) -> ArgSpec:
        """Bind concrete types to TypeVars."""
        return self.map_keys(lambda arg: _bind_free_typevars(arg, bound=bound))

    def map_keys(self, transform: Callable[[Key], Key]) -> ArgSpec:
        """Return a new ArgSpec with the keys mapped by ``callback``."""
        return ArgSpec(
            args={name: transform(arg) for name, arg in self._args.items()},
            kwargs={name: transform(arg) for name, arg in self._kwargs.items()},
            return_=self._return,
        )


@dataclass
class ProviderLocation:
    name: str
    module: str

    @classmethod
    def from_function(cls, func: ToProvider) -> ProviderLocation:
        try:
            name = func.__qualname__
        except AttributeError:
            name = str(func)
        return cls(name=name, module=_module_name(func))

    @property
    def qualname(self) -> str:
        """Fully qualified name of the provider.

        Note that this always includes the module name unlike
        ``provider.func.__qualname__`` which depends on how the provider was imported.
        """
        if self.module:
            return f'{self.module}.{self.name}'
        return self.name


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


def _module_name(x: Any) -> str:
    # getmodule might return None
    return getattr(inspect.getmodule(x), '__name__', '')


def _unwrap_decorated(func: ToProvider) -> ToProvider:
    if (wrapped := getattr(func, '__wrapped__', None)) is not None:
        return _unwrap_decorated(wrapped)
    return func


def _get_args_and_types(
    func: ToProvider,
) -> tuple[list[str], list[str], dict[str, Any]]:
    if inspect.isfunction(func):
        return _get_func_args_and_types(func)
    if inspect.ismethod(func):
        return _get_method_args_and_types(func)
    if isinstance(func, type):
        raise TypeError(
            "Classes are not supported as providers because their type annotations "
            f"cannot reliably be extracted. Got class {func}. "
            "Consider using a free function, classmethod, or staticmethod as a "
            "provider to instantiate the class. "
            "Or did you accidentally pass a class instead of an instance? "
            f"If so, use `{func}(...)` as a provider instead of `{func}`."
        ) from None
    if (call := getattr(func, '__call__', None)) is not None:  # noqa: B004
        return _get_method_args_and_types(call)
    raise TypeError(f"Cannot get type hints of provider {func}") from None


def _get_func_args_and_types(
    func: ToProvider,
) -> tuple[list[str], list[str], dict[str, Any]]:
    func = _unwrap_decorated(func)
    hints = get_type_hints(func)
    signature = inspect.getfullargspec(func)
    return signature.args, signature.kwonlyargs, hints


def _get_method_args_and_types(
    func: ToProvider,
) -> tuple[list[str], list[str], dict[str, Any]]:
    func = _unwrap_decorated(func)
    args, kwonlyargs, hints = _get_func_args_and_types(func)
    # Remove annotation of `self` / `cls` if present
    hints.pop(args[0], None)
    # Drop `self` / `cls`
    args = args[1:]
    return args, kwonlyargs, hints
