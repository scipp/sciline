# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Scope(Generic[T]):
    def __new__(cls, x) -> Any:  # type: ignore[no-untyped-def]
        return x
