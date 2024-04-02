# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

# flake8: noqa
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
from .param_table import ParamTable
from .pipeline import AmbiguousProvider, Pipeline
from .series import Series
from .task_graph import TaskGraph

__all__ = [
    "AmbiguousProvider",
    "ParamTable",
    "Pipeline",
    "scheduler",
    "Series",
    "Scope",
    "ScopeTwoParams",
    'TaskGraph',
    "UnboundTypeVar",
    "UnsatisfiedRequirement",
    "HandleAsBuildTimeException",
    "HandleAsComputeTimeException",
]

del importlib
