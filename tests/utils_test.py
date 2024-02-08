# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import sciline
from sciline import utils
from sciline.typing import Item, Label


def test_keyname_builtin() -> None:
    assert utils.keyname(int) == 'int'
    assert utils.keyname(object) == 'object'


def test_keyname_new_type() -> None:
    MyType = NewType('MyType', str)
    assert utils.keyname(MyType) == 'MyType'


def test_keyname_type_var() -> None:
    MyType = TypeVar('MyType')
    assert utils.keyname(MyType) == '~MyType'


def test_keyname_item() -> None:
    item = Item(tp=int, label=(Label(tp=float, index=0), Label(tp=str, index=1)))
    assert utils.keyname(item) == 'int(float, str)'


def test_keyname_builtin_generic() -> None:
    MyType = NewType('MyType', str)
    assert utils.keyname(list) == 'list'
    assert utils.keyname(list[float]) == 'list[float]'
    assert utils.keyname(list[MyType]) == 'list[MyType]'
    assert utils.keyname(dict[str, MyType]) == 'dict[str, MyType]'


def test_keyname_custom_generic() -> None:
    MyType = NewType('MyType', float)
    Var = TypeVar('Var')

    class G(sciline.Scope[Var, str], str):
        ...

    assert utils.keyname(G) == 'G'
    assert utils.keyname(G[int]) == 'G[int]'
    assert utils.keyname(G[MyType]) == 'G[MyType]'


def test_keyname_custom_generic_two_params() -> None:
    MyType = NewType('MyType', float)
    Var1 = TypeVar('Var1')
    Var2 = TypeVar('Var2')

    class G(sciline.ScopeTwoParams[Var1, Var2, str], str):
        ...

    assert utils.keyname(G) == 'G'
    assert utils.keyname(G[int, tuple[float]]) == 'G[int, tuple[float]]'
    assert utils.keyname(G[list[MyType], MyType]) == 'G[list[MyType], MyType]'


def test_keyqualname_builtin() -> None:
    assert utils.keyqualname(int) == 'int'
    assert utils.keyqualname(object) == 'object'


def test_keyqualname_new_type() -> None:
    # The __qualname__ of NewTypes is the same as __name__, the result is therefore
    # missing test_keyqualname_new_type.<locals>
    MyType = NewType('MyType', str)
    assert utils.keyqualname(MyType) == 'utils_test.MyType'


def test_keyqualname_type_var() -> None:
    # TypeVar has no __qualname__, the result is therefore missing
    # test_keyqualname_type_var.<locals>
    MyType = TypeVar('MyType')
    assert utils.keyqualname(MyType) == 'utils_test.~MyType'


def test_keyqualname_item() -> None:
    item = Item(tp=int, label=(Label(tp=float, index=0), Label(tp=str, index=1)))
    assert utils.keyqualname(item) == 'int(float, str)'


def test_keyqualname_builtin_generic() -> None:
    MyType = NewType('MyType', str)
    assert utils.keyqualname(list) == 'list'
    assert utils.keyqualname(list[float]) == 'list[float]'
    assert utils.keyqualname(list[MyType]) == 'list[utils_test.MyType]'
    assert utils.keyqualname(dict[str, MyType]) == 'dict[str, utils_test.MyType]'


def test_keyqualname_custom_generic() -> None:
    MyType = NewType('MyType', float)
    Var = TypeVar('Var')

    class G(sciline.Scope[Var, str], str):
        ...

    assert (
        utils.keyqualname(G) == 'utils_test.test_keyqualname_custom_generic.<locals>.G'
    )
    assert (
        utils.keyqualname(G[int])
        == 'utils_test.test_keyqualname_custom_generic.<locals>.G[int]'
    )
    assert (
        utils.keyqualname(G[MyType])
        == 'utils_test.test_keyqualname_custom_generic.<locals>.G[utils_test.MyType]'
    )


def test_keyqualname_custom_generic_two_params() -> None:
    MyType = NewType('MyType', float)
    Var1 = TypeVar('Var1')
    Var2 = TypeVar('Var2')

    class G(sciline.ScopeTwoParams[Var1, Var2, str], str):
        ...

    assert (
        utils.keyqualname(G)
        == 'utils_test.test_keyqualname_custom_generic_two_params.<locals>.G'
    )
    assert (
        utils.keyqualname(G[int, tuple[float]])
        == 'utils_test.test_keyqualname_custom_generic_two_params.'
        '<locals>.G[int, tuple[float]]'
    )
    assert (
        utils.keyqualname(G[list[MyType], MyType])
        == 'utils_test.test_keyqualname_custom_generic_two_params.<locals>.'
        'G[list[utils_test.MyType], utils_test.MyType]'
    )
