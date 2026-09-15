# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Aggregation over the rows of a table: a contribution per row, accumulators
that combine the contributions, and a finalize stage over the combined values."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any, Generic, Protocol, TypeVar

from .pipeline import Pipeline
from .scheduler import Scheduler
from .stage import Stage, warm
from .typing import Key

T = TypeVar('T')

Contribution = dict[Key, Any]
"""Values at the accumulation keys, for one member or combined."""
Table = Mapping[Hashable, Mapping[Key, Any]]
"""Rows of member-key values, by label."""


class Accumulator(Protocol[T]):
    """Combines pushed values; satisfied by any object with ``push`` and ``value``.

    Accumulators sit between the stages of an :py:class:`Aggregation`: the
    contribution of each member is pushed, the combined value is read. Whether an
    accumulator holds the pushed values or a running result is its own choice.

    :py:meth:`Aggregation.combine` pushes combined values as well as contributions,
    so that combining can proceed in groups or as a chain. For that, ``value`` must
    be pushable, and the result must not depend on how the pushes were grouped.
    """

    def push(self, value: T) -> None:
        """Add a value."""

    @property
    def value(self) -> T:
        """The combination of all values pushed so far."""


class Buffered(Generic[T]):
    """Factory for accumulators that apply an n-ary function to all pushed values.

    Each accumulator holds every pushed value until ``value`` is read, then applies
    the function to them in push order. This suits functions without a cheaper
    incremental form, such as concatenation. A sum of large arrays is better served
    by :py:class:`Reduced`.

    For combined values to be pushed back in, the function must be associative:
    ``func(func(a, b), c) == func(a, b, c)``.
    """

    def __init__(self, func: Callable[..., T]) -> None:
        """
        Parameters
        ----------
        func:
            Function combining any number of pushed values into one.
        """
        self._func = func

    def __call__(self) -> Accumulator[T]:
        """Return a new accumulator with nothing pushed."""
        return _Buffer(self._func)


class _Buffer(Generic[T]):
    def __init__(self, func: Callable[..., T]) -> None:
        self._func = func
        self._values: list[T] = []

    def push(self, value: T) -> None:
        self._values.append(value)

    @property
    def value(self) -> T:
        if not self._values:
            raise ValueError('Nothing has been pushed')
        return self._func(*self._values)


class Reduced(Generic[T]):
    """Factory for accumulators that keep a running result of a binary function.

    Each accumulator applies the function to its result so far and each pushed value,
    in push order, and holds only the result. This suits a sum of large arrays, where
    buffering would hold one array per member.

    The function must be associative, so that combining contributions in groups and
    then combining the group results gives the same value as combining them all at
    once. It must not modify its arguments: the first pushed value becomes the
    result, and pushed values are owned by the caller.
    """

    def __init__(self, func: Callable[[T, T], T]) -> None:
        """
        Parameters
        ----------
        func:
            Function combining the result so far with a pushed value into a new
            result.
        """
        self._func = func

    def __call__(self) -> Accumulator[T]:
        """Return a new accumulator with nothing pushed."""
        return _Running(self._func)


class _Running(Generic[T]):
    def __init__(self, func: Callable[[T, T], T]) -> None:
        self._func = func
        self._pushed = False
        self._result: T

    def push(self, value: T) -> None:
        self._result = self._func(self._result, value) if self._pushed else value
        self._pushed = True

    @property
    def value(self) -> T:
        if not self._pushed:
            raise ValueError('Nothing has been pushed')
        return self._result


class Aggregation:
    """Combines the contributions of the rows of a table, computed from one pipeline.

    The rows of the table are the members, its columns the *member keys*. Two
    :py:class:`Stage` objects are held: ``contribute_stage`` computes the
    *accumulation keys* from the member keys, ``finalize_stage`` computes the
    outputs from the accumulation keys. Between them, one accumulator per
    accumulation key combines the contributions of the members.

    The aggregation holds its stages, not the contributions: whoever iterates over
    the members owns them. :py:meth:`compute` is that loop for a table.
    :py:meth:`contribute`, :py:meth:`combine`, and :py:meth:`finalize` can also be
    called separately, in different processes, with the contributions serialized
    between them.

    Parameters are set on the pipeline before the aggregation is built; the
    aggregation is a snapshot of the pipeline at that time.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        members: Iterable[Key],
        accumulators: Mapping[Key, Callable[[], Accumulator[Any]]],
        outputs: Iterable[Key] = (),
        scheduler: Scheduler | None = None,
    ) -> None:
        """
        Parameters
        ----------
        pipeline:
            Pipeline with all parameters set that the outputs need, except the
            member keys.
        members:
            The keys supplied per member, the columns of the table.
        accumulators:
            Accumulation keys, each with a factory for its accumulator. Factories
            rather than instances, so that the aggregation holds no state between
            calls and whoever asks for accumulators owns their lifetime; ``combine``
            and ``compute`` use fresh ones. A key that does not depend on the members
            is not accumulated; ``finalize`` computes it from the held part of the
            graph. ``accumulation_keys`` lists the keys that are accumulated.
        outputs:
            Keys computed by ``finalize`` from the accumulation keys. Omit for an
            aggregation used only for its contributions, such as one of several
            sharing a finalize stage.
        scheduler:
            Scheduler for both stages. If not given,
            :py:class:`sciline.scheduler.DaskScheduler` is used if dask is installed,
            otherwise :py:class:`sciline.scheduler.NaiveScheduler`.

        Raises
        ------
        ValueError
            If an accumulation key or an output is not in the pipeline, or if a
            member key is not needed by the accumulation keys.
        """
        members = tuple(members)
        outputs = tuple(outputs)
        probe = Stage(pipeline, outputs=tuple(accumulators), inputs=members)
        self.accumulation_keys = probe.dynamic_outputs
        self._accumulators = {k: accumulators[k] for k in self.accumulation_keys}
        self.contribute_stage = Stage(
            pipeline,
            outputs=self.accumulation_keys,
            inputs=members,
            scheduler=scheduler,
        )
        self.finalize_stage = (
            Stage(
                pipeline,
                outputs=outputs,
                inputs=self.accumulation_keys,
                scheduler=scheduler,
            )
            if outputs
            else None
        )

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The stages of the aggregation, to warm together."""
        if self.finalize_stage is None:
            return (self.contribute_stage,)
        return (self.contribute_stage, self.finalize_stage)

    def contribute(self, row: Mapping[Key, Any]) -> Contribution:
        """Compute the contribution of one member.

        Parameters
        ----------
        row:
            A value for each member key.

        Returns
        -------
        :
            The value of each accumulation key for this member.
        """
        return self.contribute_stage.compute(row)

    def accumulators(self) -> dict[Key, Accumulator[Any]]:
        """Return new accumulators, one per accumulation key."""
        return {k: make() for k, make in self._accumulators.items()}

    def combine(self, contributions: Iterable[Contribution]) -> Contribution:
        """Push each contribution into new accumulators and read them.

        Parameters
        ----------
        contributions:
            Contributions of members, or combinations of such; see
            :py:class:`Accumulator` for what the latter asks of the accumulators.

        Returns
        -------
        :
            The combined value of each accumulation key.
        """
        acc = self.accumulators()
        for contribution in contributions:
            for key, a in acc.items():
                a.push(contribution[key])
        return {key: a.value for key, a in acc.items()}

    def finalize(self, contribution: Contribution) -> dict[Key, Any]:
        """Compute the outputs from combined contributions.

        Parameters
        ----------
        contribution:
            A value for each accumulation key.

        Returns
        -------
        :
            The value of each output key.

        Raises
        ------
        ValueError
            If the aggregation was built without outputs.
        """
        if self.finalize_stage is None:
            raise ValueError('This aggregation has no outputs')
        return self.finalize_stage.compute(contribution)

    def compute(self, table: Table) -> dict[Key, Any]:
        """Contribute per row, pushing each contribution as it is made, then finalize.

        Parameters
        ----------
        table:
            Rows of member-key values.

        Returns
        -------
        :
            The value of each output key.

        Raises
        ------
        ValueError
            If the table is empty or the aggregation was built without outputs.
        """
        if not table:
            raise ValueError('The member table is empty')
        warm(*self.stages)
        return self.finalize(
            self.combine(self.contribute(row) for row in table.values())
        )


def compute_members(
    pipeline: Pipeline, *, members: Iterable[Key], key: Key, table: Table
) -> dict[Hashable, Any]:
    """Compute a key that depends on the member keys for each row of a table.

    Parameters
    ----------
    pipeline:
        Pipeline with all parameters set that ``key`` needs, except the member keys.
    members:
        The keys supplied per member, the columns of the table.
    key:
        The key to compute.
    table:
        Rows of member-key values.

    Returns
    -------
    :
        The value of ``key`` for each row, by the row's label.
    """
    stage = Stage(pipeline, outputs=(key,), inputs=tuple(members))
    return {label: stage.compute(row)[key] for label, row in table.items()}
