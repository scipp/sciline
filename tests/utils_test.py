# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sys
from typing import NewType, TypeVar

import pytest

import sciline
from sciline import _utils
from sciline._provider import Provider
from sciline.typing import Item, Label


def module_foo(x: list[str]) -> str:
    return x[0]


def test_key_name_builtin() -> None:
    assert _utils.key_name(int) == 'int'
    assert _utils.key_name(object) == 'object'


def test_key_name_new_type() -> None:
    MyType = NewType('MyType', str)
    assert _utils.key_name(MyType) == 'MyType'


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
def test_key_name_type_alias_type() -> None:
    # Use exec to avoid a syntax error in older python
    code = """
type MyType = float
assert _utils.key_name(MyType) == 'MyType'
    """
    exec(code)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
def test_key_name_generic_type_alias_type() -> None:
    # Use exec to avoid a syntax error in older python
    code = """
type MyType[T] = dict[str, T]
assert _utils.key_name(MyType[int]) == 'MyType[int]'
    """
    exec(code)


def test_key_name_type_var() -> None:
    MyType = TypeVar('MyType')
    assert _utils.key_name(MyType) == '~MyType'  # type: ignore[arg-type]


def test_key_name_item() -> None:
    item = Item(tp=int, label=(Label(tp=float, index=0), Label(tp=str, index=1)))
    assert _utils.key_name(item) == 'int(float:0, str:1)'


def test_key_name_builtin_generic() -> None:
    MyType = NewType('MyType', str)
    assert _utils.key_name(list) == 'list'
    assert _utils.key_name(list[float]) == 'list[float]'
    assert _utils.key_name(list[MyType]) == 'list[MyType]'
    assert _utils.key_name(dict[str, MyType]) == 'dict[str, MyType]'


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_key_name_custom_generic() -> None:
    MyType = NewType('MyType', float)
    Var = TypeVar('Var')

    class G(sciline.Scope[Var, str], str):
        ...

    assert _utils.key_name(G) == 'G'
    assert _utils.key_name(G[int]) == 'G[int]'
    assert _utils.key_name(G[MyType]) == 'G[MyType]'


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_key_name_custom_generic_two_params() -> None:
    MyType = NewType('MyType', float)
    Var1 = TypeVar('Var1')
    Var2 = TypeVar('Var2')

    class G(sciline.ScopeTwoParams[Var1, Var2, str], str):
        ...

    assert _utils.key_name(G) == 'G'
    assert _utils.key_name(G[int, tuple[float]]) == 'G[int, tuple[float]]'
    assert _utils.key_name(G[list[MyType], MyType]) == 'G[list[MyType], MyType]'


def test_key_full_qualname_builtin() -> None:
    assert _utils.key_full_qualname(int) == 'builtins.int'
    assert _utils.key_full_qualname(object) == 'builtins.object'


# NewType returns a class since python 3.10,
# before that, we cannot get a proper name for it.
@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_key_full_qualname_new_type() -> None:
    # The __qualname__ of NewTypes is the same as __name__, the result is therefore
    # missing test_key_full_qualname_new_type.<locals>
    MyType = NewType('MyType', str)
    assert _utils.key_full_qualname(MyType) == 'utils_test.MyType'


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
def test_key_full_qualname_type_alias_type() -> None:
    # Use exec to avoid a syntax error in older python
    code = """
type MyType = float
assert _utils.key_full_qualname(MyType) == 'utils_test.MyType'
    """
    exec(code)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
def test_key_full_qualname_generic_type_alias_type() -> None:
    # Use exec to avoid a syntax error in older python
    code = """
type MyType[T] = dict[str, T]
assert _utils.key_full_qualname(MyType[int]) == 'utils_test.MyType[builtins.int]'
    """
    exec(code)


def test_key_full_qualname_type_var() -> None:
    # TypeVar has no __qualname__, the result is therefore missing
    # test_key_full_qualname_type_var.<locals>
    MyType = TypeVar('MyType')
    res = _utils.key_full_qualname(MyType)  # type: ignore[arg-type]
    assert res == 'utils_test.~MyType'


def test_key_full_qualname_item() -> None:
    item = Item(tp=int, label=(Label(tp=float, index=0), Label(tp=str, index=1)))
    assert (
        _utils.key_full_qualname(item)
        == 'builtins.int(builtins.float:0, builtins.str:1)'
    )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_key_full_qualname_builtin_generic() -> None:
    MyType = NewType('MyType', str)
    assert _utils.key_full_qualname(list) == 'builtins.list'
    assert _utils.key_full_qualname(list[float]) == 'builtins.list[builtins.float]'
    assert _utils.key_full_qualname(list[MyType]) == 'builtins.list[utils_test.MyType]'
    assert (
        _utils.key_full_qualname(dict[str, MyType])
        == 'builtins.dict[builtins.str, utils_test.MyType]'
    )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_key_full_qualname_custom_generic() -> None:
    MyType = NewType('MyType', float)
    Var = TypeVar('Var')

    class G(sciline.Scope[Var, str], str):
        ...

    assert (
        _utils.key_full_qualname(G)
        == 'utils_test.test_key_full_qualname_custom_generic.<locals>.G'
    )
    assert (
        _utils.key_full_qualname(G[int])
        == 'utils_test.test_key_full_qualname_custom_generic.<locals>.G[builtins.int]'
    )
    assert (
        _utils.key_full_qualname(G[MyType])
        == 'utils_test.test_key_full_qualname_custom_generic.'
        '<locals>.G[utils_test.MyType]'
    )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_key_full_qualname_custom_generic_two_params() -> None:
    MyType = NewType('MyType', float)
    Var1 = TypeVar('Var1')
    Var2 = TypeVar('Var2')

    class G(sciline.ScopeTwoParams[Var1, Var2, str], str):
        ...

    assert (
        _utils.key_full_qualname(G)
        == 'utils_test.test_key_full_qualname_custom_generic_two_params.<locals>.G'
    )
    assert (
        _utils.key_full_qualname(G[int, tuple[float]])
        == 'utils_test.test_key_full_qualname_custom_generic_two_params.'
        '<locals>.G[builtins.int, builtins.tuple[builtins.float]]'
    )
    assert (
        _utils.key_full_qualname(G[list[MyType], MyType])
        == 'utils_test.test_key_full_qualname_custom_generic_two_params.<locals>.'
        'G[builtins.list[utils_test.MyType], utils_test.MyType]'
    )


def test_provider_name_function() -> None:
    def foo(i: int) -> float:
        return float(i)

    assert _utils.provider_name(Provider.from_function(foo)) == 'foo'
    assert _utils.provider_name(Provider.from_function(module_foo)) == 'module_foo'


def test_provider_name_parameter() -> None:
    MyType = NewType('MyType', float)
    assert _utils.provider_name(Provider.parameter(4.1)) == 'parameter(float)'
    assert _utils.provider_name(Provider.parameter(MyType(3.2))) == 'parameter(float)'


def test_provider_full_qualname_function() -> None:
    def foo(i: int) -> float:
        return float(i)

    assert (
        _utils.provider_full_qualname(Provider.from_function(foo))
        == 'utils_test.test_provider_full_qualname_function.<locals>.foo'
    )
    assert (
        _utils.provider_full_qualname(Provider.from_function(module_foo))
        == 'utils_test.module_foo'
    )


def test_provider_full_qualname_parameter() -> None:
    MyType = NewType('MyType', float)
    assert _utils.provider_name(Provider.parameter(4.1)) == 'parameter(float)'
    assert _utils.provider_name(Provider.parameter(MyType(3.2))) == 'parameter(float)'
