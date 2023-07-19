# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Generic, TypeVar

T = TypeVar("T")
SUPER = TypeVar("SUPER")


class Scope(Generic[T, SUPER]):
    def __new__(cls, x: SUPER) -> SUPER:  # type: ignore[misc]
        return x
