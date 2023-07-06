# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import typing
from typing import Callable, Generic, List, Type, TypeVar, Union


def domain_type(name: str, base: type) -> type:
    class tp(base):
        pass

    return tp


def parametrized_domain_type(name: str, base: type) -> type:
    T = TypeVar('T')

    class tp(base, Generic[T]):
        pass

    return tp
