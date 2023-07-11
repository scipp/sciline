# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Generic, TypeVar

T = TypeVar("T")


class Scope(Generic[T]):
    def __new__(cls, x):  # type: ignore
        return x
