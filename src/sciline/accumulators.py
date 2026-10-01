# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Accumulators combine the values computed per member of a loop, such as the
contributions of the files of a run, before a stage computes the rest."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

T = TypeVar('T')


class Accumulator(Protocol[T]):
    """Combines pushed values; satisfied by any object with ``push`` and ``value``.

    A driver pushes the contribution of each member of a loop and reads the combined
    value. Whether an accumulator holds the pushed values or a running result is its
    own choice.

    To combine in groups or as a chain, for example in separate processes, a driver
    pushes combined values into a new accumulator. For that, ``value`` must be
    pushable, and the result must not depend on how the pushes were grouped.
    """

    def push(self, value: T) -> None:
        """Add a value."""

    @property
    def value(self) -> T:
        """The combination of all values pushed so far."""


class Buffered(Generic[T]):
    """Factory for accumulators that apply an n-ary function to all pushed values.

    Each accumulator holds every pushed value and applies the function to them, in
    push order, each time ``value`` is read. This suits functions without a cheaper
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
