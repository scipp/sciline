# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import functools
import typing
from typing import Any, Dict, Generic, NewType, TypeVar

T = TypeVar("T")


class Scope(Generic[T]):
    def __new__(cls, x):
        return x


class DomainTypeFactory:
    def __init__(self, name: str, base: type) -> None:
        self._name: str = name
        self._base: type = base
        self._subtypes: Dict[str, NewType] = {}

    def __getitem__(self, tp: type) -> type:
        return self(tp)

    def __call__(self, *args: Any, **kwargs: Any) -> NewType:
        key = f'{self._name}'
        for arg in args:
            key += f'_{arg}'
        for k, v in kwargs.items():
            key += f'_{k}_{v}'
        if (t := self._subtypes.get(key)) is not None:
            return t
        t = NewType(key, self._base)
        self._subtypes[key] = t
        return t


class SingleParameterStr(str, Generic[T]):
    def __new__(cls, x: str):
        assert isinstance(x, str)
        return x


class SingleParameterFloat(float, Generic[T]):
    def __new__(cls, x: float):
        assert isinstance(x, float)
        return x


def parametrized_domain_type(name: str, base: type) -> type:
    if base is str:

        class DomainType(SingleParameterStr[T]):
            ...

    if base is float:

        class DomainType(SingleParameterFloat[T]):
            ...

    return DomainType
    """
    Return a type-factory for parametrized domain types.

    The types return by the factory are created using typing.NewType. The returned
    factory is used similarly to a Generic, but note that the factory itself should
    not be used for annotations.

    Parameters
    ----------
    name:
        The name of the type. This is used as a prefix for the names of the types
        returned by the factory.
    base:
        The base type of the types returned by the factory.
    """
    return DomainTypeFactory(name, base)

    # class Factory:
    #    _subtypes: Dict[str, type] = {}

    #    def __class_getitem__(cls, tp: type) -> type:
    #        key = f'{name}_{tp.__name__}'
    #        if (t := cls._subtypes.get(key)) is None:
    #            t = NewType(key, base)
    #            cls._subtypes[key] = t
    #        return t

    # return Factory


_cleanups = []


def _tp_cache(func=None, /, *, typed=False):
    """Internal wrapper caching __getitem__ of generic types with a fallback to
    original function for non-hashable arguments.
    """

    def decorator(func):
        cached = functools.lru_cache(typed=typed)(func)
        _cleanups.append(cached.cache_clear)

        @functools.wraps(func)
        def inner(*args, **kwds):
            try:
                return cached(*args, **kwds)
            except TypeError:
                pass  # All real errors (not unhashable args) are raised below.
            return func(*args, **kwds)

        return inner

    if func is not None:
        return decorator(func)

    return decorator


class _Immutable:
    """Mixin to indicate that object should not be copied."""

    __slots__ = ()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class NewGenericType(_Immutable):
    def __init__(self, name, tp, *, _tvars=()):
        self.__qualname__ = name
        if '.' in name:
            name = name.rpartition('.')[-1]
        self.__name__ = name
        self.__supertype__ = tp
        self.__parameters__ = _tvars
        def_mod = typing._caller()
        if def_mod != 'typing':
            self.__module__ = def_mod

    @_tp_cache
    def __class_getitem__(cls, params):
        # copied from Generic.__class_getitem__
        if not isinstance(params, tuple):
            params = (params,)
        if not params:
            raise TypeError(
                f"Parameter list to {cls.__qualname__}[...] cannot be empty"
            )
        params = tuple(typing._type_convert(p) for p in params)
        if not all(isinstance(p, (typing.TypeVar, typing.ParamSpec)) for p in params):
            raise TypeError(
                f"Parameters to {cls.__name__}[...] must all be type variables "
                f"or parameter specification variables."
            )
        if len(set(params)) != len(params):
            raise TypeError(f"Parameters to {cls.__name__}[...] must all be unique")
        return functools.partial(cls, _tvars=params)

    @_tp_cache
    def __getitem__(self, params):
        # copied from typing.Generic.__class_getitem__
        if not isinstance(params, tuple):
            params = (params,)
        params = tuple(typing._type_convert(p) for p in params)
        if any(isinstance(t, typing.ParamSpec) for t in self.__parameters__):
            params = typing._prepare_paramspec_params(self, params)
        else:
            typing._check_generic(self, params, len(self.__parameters__))
        return typing._GenericAlias(
            self,
            params,
            _typevar_types=(typing.TypeVar, typing.ParamSpec),
            _paramspec_tvars=True,
        )

    def __repr__(self):
        return f'{self.__module__}.{self.__qualname__}'

    def __call__(self, x):
        return x

    def __reduce__(self):
        return self.__qualname__

    def __or__(self, other):
        return typing.Union[self, other]

    def __ror__(self, other):
        return typing.Union[other, self]


class MyNewType:
    def __init__(self, name, tp):
        self.__qualname__ = name
        if '.' in name:
            name = name.rpartition('.')[-1]
        self.__name__ = name
        self.__supertype__ = tp
        def_mod = typing._caller()
        if def_mod != 'typing':
            self.__module__ = def_mod

    def __repr__(self):
        return f'{self.__module__}.{self.__qualname__}'

    def __call__(self, x):
        return x

    def __reduce__(self):
        return self.__qualname__

    def __or__(self, other):
        return typing.Union[self, other]

    def __ror__(self, other):
        return typing.Union[other, self]
