# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import scheduler
from ._provider import UnboundTypeVar
from .domain import Scope, ScopeTwoParams
from .handler import (
    HandleAsBuildTimeException,
    HandleAsComputeTimeException,
    UnsatisfiedRequirement,
)
from .pipeline import Pipeline, compute_mapped, get_mapped_node_names
from .task_graph import TaskGraph

__all__ = [
    "Pipeline",
    "scheduler",
    "Scope",
    "ScopeTwoParams",
    'TaskGraph',
    "UnboundTypeVar",
    "UnsatisfiedRequirement",
    "HandleAsBuildTimeException",
    "HandleAsComputeTimeException",
    "compute_mapped",
    "get_mapped_node_names",
]

del importlib
