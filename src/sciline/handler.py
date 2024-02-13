# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import NoReturn, Protocol, TypeVar

from ._provider import ArgSpec, Provider
from .typing import Key

T = TypeVar('T')


class UnsatisfiedRequirement(Exception):
    """Raised when a type cannot be provided."""


class ErrorHandler(Protocol):
    """Error handling protocol for pipelines."""

    def handle_unsatisfied_requirement(self, tp: Key, *explanation: str) -> Provider:
        ...


class HandleAsBuildTimeException(ErrorHandler):
    """
    Error handler used by default.

    This will raise an exception when building the graph, which is helpful for
    ensuring that errors are caught early, before starting costly computation.
    """

    def handle_unsatisfied_requirement(self, tp: Key, *explanation: str) -> NoReturn:
        """Raise an exception when a type cannot be provided."""
        raise UnsatisfiedRequirement('No provider found for type', tp, *explanation)


class HandleAsComputeTimeException(ErrorHandler):
    """
    Error handler used for Pipeline.visualize().

    This avoids raising exceptions when building the graph, which would prevent
    visualization. This is helpful when visualizing a graph that is not yet complete.
    """

    def handle_unsatisfied_requirement(self, tp: Key, *explanation: str) -> Provider:
        """Return a function that raises an exception when called."""

        def unsatisfied_sentinel() -> NoReturn:
            raise UnsatisfiedRequirement('No provider found for type', tp, *explanation)

        return Provider(
            func=unsatisfied_sentinel, arg_spec=ArgSpec.null(), kind='unsatisfied'
        )
