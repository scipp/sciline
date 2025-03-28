# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Progress reporters for pipelines."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable
from time import perf_counter
from types import TracebackType
from typing import TYPE_CHECKING, Any

from ._provider import Provider
from ._utils import provider_full_qualname, provider_name

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pd = object


class Reporter(ABC):
    """Base class for progress reporters of computations.

    A ``Reporter`` instance must not be used for multiple computations concurrently!

    Reporters should be passed to :meth:`sciline.Pipeline.compute`, e.g.:

    .. code-block:: python

         from sciline import Pipeline
         from sciline.reporter import RichReporter
         reporter = RichReporter()
         pipeline = Pipeline(providers)
         pipeline.compute(Result, reporter=reporter)
    """

    def __init__(self) -> None:
        self._n_steps = 0
        self._provider_id = -1

    def run_computation(self, providers: Iterable[Provider]) -> Reporter:
        """Run a single computation with progress reporting.

        This method is primarily meant for internal use.
        As a user, you should pass reporters to :meth:`Pipeline.compute`.

        Use this method in a ``with`` statement:

        .. code-block:: python

            with reporter.run_computation(providers):
                # Perform computation

        This method exists mainly to provide arguments when entering a reporting
        context, implementers should mainly override ``__enter__``.
        """
        self._n_steps = sum(1 for provider in providers if provider.kind == "function")
        return self

    @abstractmethod
    def __enter__(self) -> None:
        """Computation is starting."""

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Computation is finished."""

    @abstractmethod
    def on_provider_start(self, provider: Provider) -> int:
        """Called when a provider is started."""

    @abstractmethod
    def on_provider_end(self, provider_id: int) -> None:
        """Called when a provider has finished."""

    def reporting_provider_func(self, provider: Provider) -> Callable[..., Any]:
        """Wrap a provider's function to report progress with this reporter."""
        if provider.kind != "function":
            return provider.func

        @functools.wraps(provider.func)
        def reporting_func(*args: Any, **kwargs: Any) -> Any:
            provider_id = self.on_provider_start(provider)
            try:
                result = provider.func(*args, **kwargs)
            finally:
                self.on_provider_end(provider_id)
            return result

        return reporting_func

    def call_provider_with_reporting(
        self, provider: Provider, values: dict[Hashable, Any]
    ) -> Any:
        """Call a provider and report its progress with this reporter."""
        if provider.kind != "function":
            return provider.call(values)

        provider_id = self.on_provider_start(provider)
        try:
            result = provider.call(values)
        finally:
            self.on_provider_end(provider_id)
        return result

    def _get_provider_id(self) -> int:
        """Return a unique identifier for a current provider."""
        self._provider_id += 1
        return self._provider_id


class TimingReporter(Reporter):
    """A reporter that tracks the time spent in each provider.

    Examples
    --------

    .. code-block:: python

        from sciline.reporter import TimingReporter
        timer = TimingReporter()
        pipeline.compute(Target, reporter=timer)
        print(timer.summary())
    """

    def __init__(self) -> None:
        super().__init__()
        self._times: dict[int, list[float | str]] = {}
        self._total_start = 0.0
        self._total_end = 0.0

    def __enter__(self) -> None:
        """Start the clock for the entire computation."""
        self._times = {}
        self._total_start = perf_counter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop the clock for the entire computation."""
        self._total_end = perf_counter()

    def on_provider_start(self, provider: Provider) -> int:
        """Start the clock for a provider call."""
        provider_id = super()._get_provider_id()
        self._times[provider_id] = [
            perf_counter(),
            0.0,
            provider_full_qualname(provider),
        ]
        return provider_id

    def on_provider_end(self, provider_id: int) -> None:
        """Stop the clock for a provider call."""
        self._times[provider_id][1] = perf_counter()

    def summary(self) -> str:
        """Return a summary of the recorded times.

        Returns
        -------
        :
            The summary as a string.

        See Also
        --------
        TimingReporter.as_pandas:
            Return the times as a Pandas DataFrame.
        """
        sorted_times = self._sorted_times()

        names = ['Provider', *(name for name, _ in sorted_times)]
        lens = _make_column(['N', *(f'({len(times)})' for _, times in sorted_times)])
        sums = _make_column(
            ['Sum [ms]', *(f'{sum(times):.3f}' for _, times in sorted_times)]
        )
        means = _make_column(
            [
                'Mean [ms]',
                *(f'{sum(times) / len(times):.3f}' for _, times in sorted_times),
            ]
        )

        return (
            f'Total time: {self._total_end - self._total_start:.3f} ms\n'
            + "\n".join(
                f'{s} {m} {ll} {n}'
                for s, m, ll, n in zip(sums, means, lens, names, strict=True)
            )
        )

    def as_pandas(self) -> pd.DataFrame:
        """Return the recorded times as a Pandas DataFrame.

        See Also
        --------
        TimingReporter.summary:
            Return a summary of the recorded times as a string.
            Does not require Pandas.
        """
        import numpy as np  # pandas depends on numpy
        import pandas as pd

        return pd.DataFrame(
            [
                (
                    name,
                    len(times),
                    np.sum(times),
                    np.min(times),
                    np.max(times),
                    np.median(times),
                    np.mean(times),
                    np.std(times),
                )
                for name, times in self._sorted_times()
            ],
            columns=[
                'Provider',
                'N Runs',
                'Sum [ms]',
                'Min [ms]',
                'Max [ms]',
                'Median [ms]',
                'Mean [ms]',
                'Std [ms]',
            ],
        )

    def _sorted_times(self) -> list[tuple[str, list[float]]]:
        """Return the recorded times in milliseconds as a sorted list."""
        run_times: dict[str, list[float]] = {}
        for start, end, name in self._times.values():
            run_times.setdefault(name, []).append((end - start) * 1000)  # type: ignore[arg-type, operator]
        return sorted(run_times.items(), key=lambda x: sum(x[1]), reverse=True)


def _make_column(values: list[str]) -> list[str]:
    width = max(len(v) for v in values)
    return [f'{v:>{width}}' for v in values]


class RichReporter(Reporter):
    """Report progress using Rich.

    This class uses the `rich <https://rich.readthedocs.io/en/stable/index.html>`_
    package to display a progress bar.
    Rich is not a hard dependency of Sciline and must be installed separately.

    Examples
    --------

    .. code-block:: python

         from sciline import Pipeline
         from sciline.reporter import RichReporter
         reporter = RichReporter()
         pipeline = Pipeline(providers)
         pipeline.compute(Result, reporter=reporter)
    """

    def __init__(
        self,
        *,
        description: str = "Computing",
        show_remaining_time: bool = False,
        **rich_progress_args: Any,
    ) -> None:
        """Initialize a RichReporter.

        Parameters
        ----------
        description:
            Description shown with the progress bar.
        show_remaining_time:
            Show an estimate of the remaining time.
            This is off by default because providers tend to have very different
            running times and the estimate is unlikely to be useful.
        rich_progress_args:
            Additional arguments passed to the
            `rich.Progress <https://rich.readthedocs.io/en/stable/reference/progress.html#rich.progress.Progress>`_
            constructor.
        """

        from rich import progress

        super().__init__()
        self._description = description
        self._show_remaining_time = show_remaining_time
        self._progress_args = rich_progress_args

        self._active_provider_list = _ProviderList()
        self._progress_bar = _make_rich_progress_bar(
            active_provider_list=self._active_provider_list,
            show_remaining_time=show_remaining_time,
            **rich_progress_args,
        )
        self._task_id = progress.TaskID(0)

    def __enter__(self) -> None:
        """Computation is starting."""
        self._task_id = self._progress_bar.add_task(
            self._description, total=self._n_steps
        )
        self._progress_bar.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Computation is finished."""
        self._progress_bar.__exit__(exc_type, exc_val, exc_tb)

    def on_provider_start(self, provider: Provider) -> int:
        """Show that a provider has started."""
        provider_id = self._get_provider_id()
        self._active_provider_list.add_provider(provider_id, provider)
        return provider_id

    def on_provider_end(self, provider_id: int) -> None:
        """Show that a provider has finished."""
        self._active_provider_list.remove_provider(provider_id)
        self._progress_bar.update(self._task_id, advance=1)


def _make_rich_progress_bar(
    active_provider_list: _ProviderList,
    show_remaining_time: bool = False,
    **rich_progress_args: Any,
) -> Any:
    from rich import progress

    columns = [
        progress.TextColumn("[progress.description]{task.description}"),
        progress.BarColumn(),
        progress.MofNCompleteColumn(),
        progress.TimeElapsedColumn(),
    ]
    if show_remaining_time:
        columns.extend(
            (
                progress.RenderableColumn("|"),
                progress.TimeRemainingColumn(),
            )
        )
    columns.append(progress.RenderableColumn(active_provider_list))
    progress_bar = progress.Progress(*columns, **rich_progress_args)
    return progress_bar


class NullReporter(Reporter):
    """A progress reporter that does nothing.

    Useful as a fallback to disable reporting.
    """

    def __enter__(self) -> None:
        """Computation is starting."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Computation is finished."""

    def on_provider_start(self, provider: Provider) -> int:
        """Show that a provider has started."""
        return self._get_provider_id()

    def on_provider_end(self, provider_id: int) -> None:
        """Show that a provider has finished."""

    def reporting_provider_func(self, provider: Provider) -> Callable[..., Any]:
        """Call a provider and report its progress with this reporter."""
        # Override base method to avoid overhead.
        return provider.func

    def call_provider_with_reporting(
        self, provider: Provider, values: dict[Hashable, Any]
    ) -> Any:
        """Call a provider and report its progress with this reporter."""
        # Override base method to avoid overhead.
        return provider.call(values)


class _ProviderList:
    """Track the names of currently active providers.

    This class is a
    `rich.console.Renderable https://rich.readthedocs.io/en/stable/reference/console.html#rich.console.ConsoleRenderable`
    that displays the short names of providers.
    """

    def __init__(self) -> None:
        self._provider_names: dict[int, str] = {}

    def add_provider(self, provider_id: int, provider: Provider) -> None:
        self._provider_names[provider_id] = provider_name(provider)

    def remove_provider(self, provider_id: int) -> None:
        del self._provider_names[provider_id]

    def __rich_console__(self, console: Any, options: Any) -> Any:
        if self._provider_names:
            active_providers = rf"\[{', '.join(self._provider_names.values())}]"
        else:
            active_providers = ""
        # Not overflowing the line seems more defensive when this code is used
        # in a larger context.
        options.overflow = 'ellipsis'
        options.no_wrap = True
        return console.render(active_providers, options)
