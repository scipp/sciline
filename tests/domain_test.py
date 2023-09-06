# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import pytest

import sciline as sl

T = TypeVar("T")


def test_mypy_detects_wrong_arg_type_of_Scope_subclass() -> None:
    Param = TypeVar('Param')
    Param1 = NewType('Param1', int)

    class A(sl.Scope[Param, float], float):
        ...

    A[Param1](1.5)
    A[Param1]('abc')  # type: ignore[arg-type]


def test_missing_interface_of_scope_subclass_raises() -> None:
    param = TypeVar('param')

    with pytest.raises(TypeError, match="Missing or wrong interface for"):

        class A(sl.Scope[param, float]):
            ...


def test_mypy_accepts_interface_of_scope_sibling_class() -> None:
    param = TypeVar('param')
    param1 = NewType('param1', int)

    class A(sl.Scope[param, float], float):
        ...

    a = A[param1](1.5)
    a + a


def test_Scope_inconsistent_type_and_interface_raises() -> None:
    param = TypeVar('param')

    with pytest.raises(TypeError, match="Missing or wrong interface for"):

        class A(sl.Scope[param, str], float):
            ...


def test_ScopeTwoParams_inconsistent_type_and_interface_raises() -> None:
    Param1 = TypeVar('Param1')
    Param2 = TypeVar('Param2')

    with pytest.raises(TypeError, match="Missing or wrong interface for"):

        class A(sl.ScopeTwoParams[Param1, Param2, str], float):
            ...


def test_ScopeTwoParams() -> None:
    Param1 = TypeVar('Param1')
    Param2 = TypeVar('Param2')

    class A(sl.ScopeTwoParams[Param1, Param2, float], float):
        ...

    assert isinstance(A[int, int](1.5), float)
