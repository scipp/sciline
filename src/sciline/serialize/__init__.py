# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Serialization of task graphs.

.. autosummary::
    :toctree: ../generated/functions
    :template: function-template.rst

    json_schema
    json_serialize_task_graph
"""

from ._json import json_schema, json_serialize_task_graph

__all__ = ['json_schema', 'json_serialize_task_graph']
