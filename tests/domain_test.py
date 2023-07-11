# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sciline as sl
import typing


def test_domain_type_factory():
    Str = sl.DomainTypeFactory('Str', str)
    tp = Str('a', b=1)
    assert tp is tp
    assert tp == tp


def test_NewGenericType():
    T = typing.TypeVar('T')
    Str = sl.domain.NewGenericType[T]('Str', str)
    assert Str[int] == Str[int]
