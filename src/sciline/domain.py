# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Generic, TypeVar

PARAM = TypeVar("PARAM")
SUPER = TypeVar("SUPER")


class Scope(Generic[PARAM, SUPER]):
    """
    Helper for defining a generic type alias.
    """

    def __new__(cls, x: SUPER) -> SUPER:  # type: ignore[misc]
        return x
