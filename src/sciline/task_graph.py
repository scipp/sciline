# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Optional, Tuple, Union

from sciline.scheduler import DaskScheduler, Graph, NaiveScheduler, Scheduler


class TaskGraph:
    def __init__(
        self,
        *,
        graph: Graph,
        keys: Union[type, Tuple[type, ...]],
        scheduler: Optional[Scheduler] = None,
    ) -> None:
        self._graph = graph
        self._keys = keys
        if scheduler is None:
            try:
                scheduler = DaskScheduler()
            except ImportError:
                scheduler = NaiveScheduler()
        self._scheduler = scheduler

    def compute(self, keys: Optional[Union[type, Tuple[type, ...]]] = None) -> Any:
        """
        Compute the result of the graph.
        Parameters
        ----------
        keys:
            Optional list of keys to compute. This can be used to override the keys
            stored in the graph instance. Note that the keys must be present in the
            graph as intermediate results, otherwise KeyError is raised.
        """
        if keys is None:
            keys = self._keys
        if isinstance(keys, tuple):
            return self._scheduler.get(self._graph, list(keys))
        else:
            return self._scheduler.get(self._graph, [keys])[0]