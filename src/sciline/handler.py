# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Callable, Protocol, Type, TypeVar, Union

from .typing import Item

T = TypeVar('T')


class UnsatisfiedRequirement(Exception):
    """Raised when a type cannot be provided."""


class Handler(Protocol):
    """Error handling protocol for pipelines."""

    def handle_unsatisfied_requirement(self, tp: Union[Type[T], Item[T]]):
        ...


class HandleAsBuildTimeException(Handler):
    """
    Error handler used by default.

    This will raise an exception when building the graph, which is helpful for
    ensuring that errors are caught early, before starting costly computation.
    """

    def handle_unsatisfied_requirement(self, tp: Union[Type[T], Item[T]]):
        """Raise an exception when a type cannot be provided."""
        raise UnsatisfiedRequirement('No provider found for type', tp)


class HandleAsComputeTimeException(Handler):
    """
    Error handler used for Pipeline.visualize().

    This avoids raising exceptions when building the graph, which would prevent
    visualization. This is helpful when visualizing a graph that is not yet complete.
    """

    def handle_unsatisfied_requirement(
        self, tp: Union[Type[T], Item[T]]
    ) -> Callable[[], T]:
        """Return a function that raises an exception when called."""

        def unsatisfied_sentinel() -> None:
            raise UnsatisfiedRequirement('No provider found for type', tp)

        return unsatisfied_sentinel
