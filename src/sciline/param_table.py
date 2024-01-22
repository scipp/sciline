# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any, Collection, Dict, Mapping, Optional


class ParamTable(Mapping[type, Collection[Any]]):
    """A table of parameters with a row index and named row dimension."""

    def __init__(
        self,
        row_dim: type,
        columns: Dict[type, Collection[Any]],
        *,
        index: Optional[Collection[Any]] = None,
    ):
        """
        Create a new param table.

        Parameters
        ----------
        row_dim:
            The row dimension. This must be a type or a type-alias (not an instance),
            and is used by :py:class:`sciline.Pipeline` to identify each parameter
            table.
        columns:
            The columns of the table. The keys (column names) must be types or type-
            aliases matching the values in the respective columns.
        index:
            The row index of the table. If not given, a default index will be
            generated, as the integer range of the column length.
        """
        sizes = set(len(v) for v in columns.values())
        if len(sizes) > 1:
            raise ValueError(
                f"Columns in param table must all have same size, got {sizes}"
            )
        if sizes:
            size = sizes.pop()
        elif index is None:
            raise ValueError("Cannot create param table with zero columns and no index")
        else:
            size = len(index)
        if index is not None:
            if len(index) != size:
                raise ValueError(
                    f"Index length not matching columns, got {len(index)} and {size}"
                )
            if len(set(index)) != len(index):
                raise ValueError(f"Index must be unique, got {index}")
        self._row_dim = row_dim
        self._columns = columns
        self._index = index or list(range(size))

    @property
    def row_dim(self) -> type:
        """The row dimension of the table."""
        return self._row_dim

    @property
    def index(self) -> Collection[Any]:
        """The row index of the table."""
        return self._index

    def __contains__(self, key: Any) -> bool:
        return self._columns.__contains__(key)

    def __getitem__(self, key: Any) -> Any:
        return self._columns.__getitem__(key)

    def __iter__(self) -> Any:
        return self._columns.__iter__()

    def __len__(self) -> int:
        return self._columns.__len__()

    def __repr__(self) -> str:
        return f"ParamTable(row_dim={self.row_dim.__name__}, columns={self._columns})"

    def _repr_html_(self) -> str:
        return (
            f"<table><tr><th>{self.row_dim.__name__}</th>"
            + "".join(
                f"<th>{getattr(k, '__name__', str(k).split('.')[-1])}</th>"
                for k in self._columns.keys()
            )
            + "</tr>"
            + "".join(
                f"<tr><td>{idx}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
                for idx, row in zip(self.index, zip(*self._columns.values()))
            )
            + "</table>"
        )
