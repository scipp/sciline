# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
# ruff: noqa: RUF100, E402, F401, I

import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import accumulators, scheduler
from ._provider import Provider, UnboundTypeVar
from .domain import Scope
from .handler import (
    HandleAsBuildTimeException,
    HandleAsComputeTimeException,
    UnsatisfiedRequirement,
)
from .pipeline import Pipeline, compute_mapped, get_mapped_node_names
from .stage import Stage, prepare_stages, visualize_stages
from .task_graph import TaskGraph

__all__ = [
    "HandleAsBuildTimeException",
    "HandleAsComputeTimeException",
    "Pipeline",
    "Provider",
    "Scope",
    "Stage",
    'TaskGraph',
    "UnboundTypeVar",
    "UnsatisfiedRequirement",
    "accumulators",
    "compute_mapped",
    "get_mapped_node_names",
    "prepare_stages",
    "scheduler",
    "visualize_stages",
]

del importlib
