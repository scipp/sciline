# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import TypeVar

from ._provider import Provider

T = TypeVar('T')


Key = type
Graph = dict[Key, Provider]


Json = dict[str, "Json"] | list["Json"] | str | int | float | bool | None
