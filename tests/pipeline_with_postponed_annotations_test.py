# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Test Pipeline with postponed evaluation of type hints.

Sciline works with postponed evaluation for globally defined types.
But ``typing.get_type_hints`` cannot handle locally defined types
(i.e., defined in a function).
There is nothing we can do about it, so these tests just check that
everything behaves as expected.
"""
from __future__ import annotations

from typing import NewType, TypeVar

import pytest

import sciline as sl

Int = NewType('Int', int)
Str = NewType('Str', str)
T = TypeVar('T')


class Number(sl.Scope[T, int], int):
    ...


class BigNumber(sl.Scope[T, int], int):
    ...


def test_postponed_annotation_builtin() -> None:
    def to_string(x: int) -> str:
        return str(x)

    pl = sl.Pipeline([to_string], params={int: 6})
    assert pl.compute(str) == '6'


def test_postponed_annotation_new_type() -> None:
    def to_string(x: Int) -> Str:
        return Str(str(x))

    pl = sl.Pipeline([to_string], params={Int: 6})
    assert pl.compute(Str) == '6'


def test_postponed_annotation_local_new_type() -> None:
    LInt = NewType('LInt', int)
    LStr = NewType('LStr', str)

    def to_string(x: LInt) -> LStr:
        return LStr(str(x))

    # typing.get_type_hints cannot find LInt and LStr.
    with pytest.raises(NameError):
        sl.Pipeline([to_string], params={LInt: 6})


def test_postponed_annotation_generic_builtin() -> None:
    def lengths(lst: list[str]) -> list[int]:
        return [len(s) for s in lst]

    pl = sl.Pipeline([lengths], params={list[str]: ['abc', 'de', 'f']})
    assert pl.compute(list[int]) == [3, 2, 1]


def test_postponed_annotation_generic_custom() -> None:
    def double(n: Number[T]) -> BigNumber[T]:
        return BigNumber[T](n * 2)

    pl = sl.Pipeline([double], params={Number[Int]: 3, Number[Str]: 5})
    assert pl.compute(BigNumber[Int]) == BigNumber[Int](6)
    assert pl.compute(BigNumber[Str]) == BigNumber[Str](10)


def test_postponed_annotation_generic_local_custom() -> None:
    LT = TypeVar('LT')

    class LNumber(sl.Scope[LT, int], int):
        ...

    class LBigNumber(sl.Scope[LT, int], int):
        ...

    def double(n: LNumber[LT]) -> LBigNumber[LT]:
        return LBigNumber[LT](n * 2)

    # typing.get_type_hints cannot find LNumber and LBigNumber.
    with pytest.raises(NameError):
        sl.Pipeline([double], params={LNumber[Int]: 3, LNumber[Str]: 5})
