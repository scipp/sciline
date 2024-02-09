# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sys
from typing import NewType, TypeVar

import pytest

import sciline
from sciline import utils
from sciline.typing import Item, Label


def test_keyname_builtin() -> None:
    assert utils.keyname(int) == 'int'
    assert utils.keyname(object) == 'object'


def test_keyname_new_type() -> None:
    MyType = NewType('MyType', str)
    assert utils.keyname(MyType) == 'MyType'


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
def test_keyname_type_alias_type() -> None:
    # Use exec to avoid a syntax error in older python
    code = """
type MyType = float
assert utils.keyname(MyType) == 'MyType'
    """
    exec(code)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
def test_keyname_generic_type_alias_type() -> None:
    # Use exec to avoid a syntax error in older python
    code = """
type MyType[T] = dict[str, T]
assert utils.keyname(MyType[int]) == 'MyType[int]'
    """
    exec(code)


def test_keyname_type_var() -> None:
    MyType = TypeVar('MyType')
    assert utils.keyname(MyType) == 'MyType'


def test_keyname_item() -> None:
    item = Item(tp=int, label=(Label(tp=float, index=0), Label(tp=str, index=1)))
    assert utils.keyname(item) == 'int(float, str)'


def test_keyname_builtin_generic() -> None:
    MyType = NewType('MyType', str)
    assert utils.keyname(list) == 'list'
    assert utils.keyname(list[float]) == 'list[float]'
    assert utils.keyname(list[MyType]) == 'list[MyType]'
    assert utils.keyname(dict[str, MyType]) == 'dict[str, MyType]'


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_keyname_custom_generic() -> None:
    MyType = NewType('MyType', float)
    Var = TypeVar('Var')

    class G(sciline.Scope[Var, str], str):
        ...

    assert utils.keyname(G) == 'G'
    assert utils.keyname(G[int]) == 'G[int]'
    assert utils.keyname(G[MyType]) == 'G[MyType]'


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_keyname_custom_generic_two_params() -> None:
    MyType = NewType('MyType', float)
    Var1 = TypeVar('Var1')
    Var2 = TypeVar('Var2')

    class G(sciline.ScopeTwoParams[Var1, Var2, str], str):
        ...

    assert utils.keyname(G) == 'G'
    assert utils.keyname(G[int, tuple[float]]) == 'G[int, tuple[float]]'
    assert utils.keyname(G[list[MyType], MyType]) == 'G[list[MyType], MyType]'


def test_keyfullqualname_builtin() -> None:
    assert utils.keyfullqualname(int) == 'int'
    assert utils.keyfullqualname(object) == 'object'


# NewType returns a class since python 3.10,
# before that, we cannot get a proper name for it.
@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_keyfullqualname_new_type() -> None:
    # The __qualname__ of NewTypes is the same as __name__, the result is therefore
    # missing test_keyfullqualname_new_type.<locals>
    MyType = NewType('MyType', str)
    assert utils.keyfullqualname(MyType) == 'utils_test.MyType'


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
def test_keyfullqualname_type_alias_type() -> None:
    # Use exec to avoid a syntax error in older python
    code = """
type MyType = float
assert utils.keyfullqualname(MyType) == 'utils_test.MyType'
    """
    exec(code)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
def test_keyfullqualname_generic_type_alias_type() -> None:
    # Use exec to avoid a syntax error in older python
    code = """
type MyType[T] = dict[str, T]
assert utils.keyfullqualname(MyType[int]) == 'utils_test.MyType[int]'
    """
    exec(code)


def test_keyfullqualname_type_var() -> None:
    # TypeVar has no __qualname__, the result is therefore missing
    # test_keyfullqualname_type_var.<locals>
    MyType = TypeVar('MyType')
    assert utils.keyfullqualname(MyType) == 'utils_test.~MyType'


def test_keyfullqualname_item() -> None:
    item = Item(tp=int, label=(Label(tp=float, index=0), Label(tp=str, index=1)))
    assert utils.keyfullqualname(item) == 'int(float, str)'


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_keyfullqualname_builtin_generic() -> None:
    MyType = NewType('MyType', str)
    assert utils.keyfullqualname(list) == 'list'
    assert utils.keyfullqualname(list[float]) == 'list[float]'
    assert utils.keyfullqualname(list[MyType]) == 'list[utils_test.MyType]'
    assert utils.keyfullqualname(dict[str, MyType]) == 'dict[str, utils_test.MyType]'


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_keyfullqualname_custom_generic() -> None:
    MyType = NewType('MyType', float)
    Var = TypeVar('Var')

    class G(sciline.Scope[Var, str], str):
        ...

    assert (
        utils.keyfullqualname(G)
        == 'utils_test.test_keyfullqualname_custom_generic.<locals>.G'
    )
    assert (
        utils.keyfullqualname(G[int])
        == 'utils_test.test_keyfullqualname_custom_generic.<locals>.G[int]'
    )
    assert (
        utils.keyfullqualname(G[MyType])
        == 'utils_test.test_keyfullqualname_custom_generic.'
        '<locals>.G[utils_test.MyType]'
    )


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_keyfullqualname_custom_generic_two_params() -> None:
    MyType = NewType('MyType', float)
    Var1 = TypeVar('Var1')
    Var2 = TypeVar('Var2')

    class G(sciline.ScopeTwoParams[Var1, Var2, str], str):
        ...

    assert (
        utils.keyfullqualname(G)
        == 'utils_test.test_keyfullqualname_custom_generic_two_params.<locals>.G'
    )
    assert (
        utils.keyfullqualname(G[int, tuple[float]])
        == 'utils_test.test_keyfullqualname_custom_generic_two_params.'
        '<locals>.G[int, tuple[float]]'
    )
    assert (
        utils.keyfullqualname(G[list[MyType], MyType])
        == 'utils_test.test_keyfullqualname_custom_generic_two_params.<locals>.'
        'G[list[utils_test.MyType], utils_test.MyType]'
    )
