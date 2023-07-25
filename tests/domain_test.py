# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, TypeVar

import sciline as sl

T = TypeVar("T")


def test_mypy_detects_wrong_arg_type_of_Scope_subclass() -> None:
    Param = TypeVar('Param')
    Param1 = NewType('Param1', int)

    class A(sl.Scope[Param, float]):
        ...

    A[Param1](1.5)
    A[Param1]('abc')  # type: ignore[arg-type]


def test_mypy_detects_missing_interface_of_scope_subclass() -> None:
    param = TypeVar('param')
    param1 = NewType('param1', int)

    class A(sl.Scope[param, float]):
        ...

    a = A[param1](1.5)
    a + a  # type: ignore[operator]


def test_mypy_accepts_interface_of_scope_sibling_class() -> None:
    param = TypeVar('param')
    param1 = NewType('param1', int)

    class A(sl.Scope[param, float], float):
        ...

    a = A[param1](1.5)
    a + a


def test_mypy_partially_detects_inconsistent_type_and_interface() -> None:
    param = TypeVar('param')
    param1 = NewType('param1', int)

    # We will get a str instance but tell mypy (via inheritance) that it is a float.
    # This is not directly detected by mypy, but we may get helpful errors later on
    # when we use the instance in incompatible ways.
    class A(sl.Scope[param, str], float):
        ...

    a = A[param1]('abc')
    a.capitalize()  # type: ignore[attr-defined]
