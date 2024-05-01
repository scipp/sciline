# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from types import UnionType
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
    overload,
)

from ._provider import Provider, ToProvider
from .data_graph import DataGraph
from .display import pipeline_html_repr
from .handler import ErrorHandler, HandleAsComputeTimeException
from .scheduler import Scheduler
from .task_graph import TaskGraph
from .typing import Item, Key

T = TypeVar('T')
KeyType = TypeVar('KeyType', bound=Key)


def _is_multiple_keys(keys: type | Iterable[type]) -> bool:
    # Cannot simply use isinstance(keys, Iterable) because that is True for
    # generic aliases of iterable types, e.g.,
    #
    # class Str(sl.Scope[Param, str], str): ...
    # keys = Str[int]
    #
    # And isinstance(keys, type) does not work on its own because
    # it is False for the above type.
    return (
        not isinstance(keys, type) and not get_args(keys) and isinstance(keys, Iterable)
    )


class Pipeline(DataGraph):
    """A container for providers that can be assembled into a task graph."""

    def __init__(
        self,
        providers: Optional[Iterable[Union[ToProvider, Provider]]] = None,
        *,
        params: Optional[Dict[Type[Any], Any]] = None,
    ):
        """
        Setup a Pipeline from a list providers

        Parameters
        ----------
        providers:
            List of callable providers. Each provides its return value.
            Their arguments and return value must be annotated with type hints.
        params:
            Dictionary of concrete values to provide for types.
        """
        super().__init__(providers)
        for tp, param in (params or {}).items():
            self[tp] = param

    @overload
    def compute(self, tp: Type[T], **kwargs: Any) -> T:
        ...

    @overload
    def compute(self, tp: Iterable[Type[T]], **kwargs: Any) -> Dict[Type[T], T]:
        ...

    @overload
    def compute(self, tp: Item[T], **kwargs: Any) -> T:
        ...

    @overload
    def compute(self, tp: UnionType, **kwargs: Any) -> Any:
        ...

    def compute(
        self, tp: type | Iterable[type] | Item[T] | UnionType, **kwargs: Any
    ) -> Any:
        """
        Compute result for the given keys.

        Equivalent to ``self.get(tp).compute()``.

        Parameters
        ----------
        tp:
            Type to compute the result for.
            Can be a single type or an iterable of types.
        kwargs:
            Keyword arguments passed to the ``.get()`` method.
        """
        return self.get(tp, **kwargs).compute()

    def visualize(
        self, tp: type | Iterable[type], **kwargs: Any
    ) -> graphviz.Digraph:  # type: ignore[name-defined] # noqa: F821
        """
        Return a graphviz Digraph object representing the graph for the given keys.

        Equivalent to ``self.get(tp).visualize()``.

        Parameters
        ----------
        tp:
            Type to visualize the graph for.
            Can be a single type or an iterable of types.
        kwargs:
            Keyword arguments passed to :py:class:`graphviz.Digraph`.
        """
        return self.get(tp, handler=HandleAsComputeTimeException()).visualize(**kwargs)

    def get(
        self,
        keys: type | Iterable[type] | Item[T] | object,
        *,
        scheduler: Optional[Scheduler] = None,
        handler: Optional[ErrorHandler] = None,
    ) -> TaskGraph:
        """
        Return a TaskGraph for the given keys.

        Parameters
        ----------
        keys:
            Type to compute the result for.
            Can be a single type or an iterable of types.
        scheduler:
            Optional scheduler to use for computing the result. If not given, a
            :py:class:`NaiveScheduler` is used if `dask` is not installed,
            otherwise dask's threaded scheduler is used.
        handler:
            Handler for unsatisfied requirements. If not provided,
            :py:class:`HandleAsBuildTimeException` is used, which raises an exception.
            During development and debugging it can be helpful to use a handler that
            raises an exception only when the graph is computed. This can be achieved
            by passing :py:class:`HandleAsComputeTimeException` as the handler.
        """
        if multi := _is_multiple_keys(keys):
            targets = tuple(keys)  # type: ignore[arg-type]
        else:
            targets = (keys,)
        graph = self.build(targets=targets, handler=handler)
        return TaskGraph(
            graph=graph,
            targets=targets if multi else keys,
            scheduler=scheduler,
        )

    @overload
    def bind_and_call(self, fns: Callable[..., T], /) -> T:
        ...

    @overload
    def bind_and_call(self, fns: Iterable[Callable[..., Any]], /) -> Tuple[Any, ...]:
        ...

    def bind_and_call(
        self, fns: Union[Callable[..., Any], Iterable[Callable[..., Any]]], /
    ) -> Any:
        """
        Call the given functions with arguments provided by the pipeline.

        Parameters
        ----------
        fns:
            Functions to call.
            The pipeline will provide all arguments based on the function's type hints.

            If this is a single callable, it is called directly.
            Otherwise, ``bind_and_call`` will iterate over it and call all functions.
            If will in either case call :meth:`Pipeline.compute` only once.

        Returns
        -------
        :
            The return values of the functions in the same order as the functions.
            If only one function is passed, its return value
            is *not* wrapped in a tuple.
        """
        return_tuple = True
        if callable(fns):
            fns = (fns,)
            return_tuple = False

        arg_types_per_function = {
            fn: {
                name: ty for name, ty in get_type_hints(fn).items() if name != 'return'
            }
            for fn in fns
        }
        all_arg_types = tuple(
            set(chain(*(a.values() for a in arg_types_per_function.values())))
        )
        values_per_type = self.compute(all_arg_types)
        results = tuple(
            fn(**{name: values_per_type[ty] for name, ty in arg_types.items()})
            for fn, arg_types in arg_types_per_function.items()
        )
        if not return_tuple:
            return results[0]
        return results

    def __copy__(self) -> Pipeline:
        return self.copy()

    def _repr_html_(self) -> str:
        providers_without_parameters = (
            (origin, tuple(), value) for origin, value in self._providers.items()
        )  # type: ignore[var-annotated]
        providers_with_parameters = (
            (origin, args, value)
            for origin in self._subproviders
            for args, value in self._subproviders[origin].items()
        )
        return pipeline_html_repr(
            chain(providers_without_parameters, providers_with_parameters)
        )
