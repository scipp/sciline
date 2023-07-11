# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import typing

import numpy as np

import sciline as sl


def factory() -> typing.Callable:
    T = typing.TypeVar('T')

    def func(x: T) -> typing.List[T]:
        return [x, x]

    return func


def test_factory():
    f = factory()
    typing.get_type_hints(f)['return']
    assert f(1) == [1, 1]


def test_providers():
    providers = {}
    f = factory()
    for tp in (int, float):
        generic = typing.get_type_hints(f)['return']
        # replace the typevar with the concrete type
        concrete = generic[tp]
        providers[concrete] = f

    def func(arg: typing.List[int]) -> int:
        return sum(arg)

    assert func(providers.get(typing.get_type_hints(func)['arg'])(1)) == 2


T = typing.TypeVar('T')


# Str = sl.domain.MyNewType('Str', str)
# Str = typing.NewType('Str', str)


class Str(typing.Generic[T]):
    def __init__(self, a):
        self.__a = a

    def __getattr__(self, attr):
        return getattr(self.__a, attr)

    def __setattr__(self, attr, val):
        if attr == '_Str__a':
            object.__setattr__(self, attr, val)

        return setattr(self.__a, attr, val)


def str_factory() -> typing.Callable:
    T = typing.TypeVar('T')

    def func(x: T) -> Str[T]:
        return Str(f'{x}')

    return func


def test_Str():
    f = str_factory()
    assert f(1) == '1'


def make_domain_type(
    base, types: typing.Tuple[type, ...]
) -> typing.Dict[type, typing.NewType]:
    return {tp: typing.NewType(f'IofQ_{tp.__name__}', base) for tp in types}


def test_make_domain_type():
    Raw = make_domain_type(list, (int, float))
    IofQ = make_domain_type(str, (int, float))

    T = typing.TypeVar('T')

    def func(x: Raw[T]) -> IofQ[T]:
        return f'{x}'

    assert func(Raw[int]([1, 2])) == '[1, 2]'
    assert func(1.0) == '1.0'


def test_wrapping():
    T = typing.TypeVar('T')

    class Raw(typing.Generic[T]):
        def __init__(self, value: list):
            self.value: list = value

    class IofQ(typing.Generic[T]):
        def __init__(self, value: str):
            self.value: str = value

    def factory(tp: type) -> typing.Callable:
        class DomainType(typing.Generic[T]):
            def __init__(self, value: tp):
                self.value: str = value

    def func(x: Raw[T]) -> IofQ[T]:
        return IofQ[T](f'{x.value}')

    assert func(Raw[int]([1])).value == '[1]'
    assert func(Raw[float]([1.0])).value == '[1.0]'


from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)


class SingleParameterGeneric(Generic[T]):
    def __new__(cls, x: np.ndarray):
        assert isinstance(x, np.ndarray)
        return x


class DomainType(SingleParameterGeneric[T], np.ndarray):
    ...


class AnotherDomainType(SingleParameterGeneric[T], np.ndarray):
    ...


DataType = TypeVar("DataType")


def foo(data: DomainType[DataType]) -> AnotherDomainType[DataType]:
    return AnotherDomainType(data + 1)


def test_foo() -> None:
    assert np.array_equal(foo(DomainType(np.array([1, 2, 3]))), [1, 2, 3])
    a = np.array([1, 2, 3])
    assert DomainType(a) is a

    Array = typing.NewType("Array", np.ndarray)
    assert Array(a) is a


def func(x: AnotherDomainType[int]) -> int:
    return np.sum(x)


def make_int() -> DomainType[int]:
    return DomainType(np.array([1, 2, 3]))


def test_injection() -> None:
    providers: Dict[type, Callable[..., Any]] = {
        int: func,
        DomainType[int]: make_int,
        AnotherDomainType: foo,
    }

    Return = typing.TypeVar("Return")

    def call(func: Callable[..., Return], bound: Optional[Any] = None) -> Return:
        tps = get_type_hints(func)
        del tps['return']
        args: Dict[str, Any] = {}
        for name, tp in tps.items():
            if (provider := providers.get(tp)) is not None:
                args[name] = call(provider, bound)
            elif (origin := get_origin(tp)) is not None:
                if (provider := providers.get(origin)) is not None:
                    args[name] = call(provider, *get_args(tp))
                else:
                    provider = providers[origin[bound]]
                    args[name] = call(provider, bound)
        return func(**args)

    assert call(func) == 9
