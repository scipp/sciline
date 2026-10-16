# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
# ruff: noqa: RUF100, E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import scheduler
from ._provider import Provider, UnboundTypeVar
from .domain import NativeScope, Scope, native_scope
from .handler import (
    HandleAsBuildTimeException,
    HandleAsComputeTimeException,
    UnsatisfiedRequirement,
)
from .pipeline import Pipeline, compute_mapped, get_mapped_node_names
from .task_graph import TaskGraph

__all__ = [
    "HandleAsBuildTimeException",
    "HandleAsComputeTimeException",
    "NativeScope",
    "Pipeline",
    "Provider",
    "Scope",
    'TaskGraph',
    "UnboundTypeVar",
    "UnsatisfiedRequirement",
    "compute_mapped",
    "get_mapped_node_names",
    "native_scope",
    "scheduler",
]

del importlib
