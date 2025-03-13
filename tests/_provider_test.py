# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# Tests on the hidden provider module

from typing import TypeVar

from sciline._provider import ArgSpec


def test_arg_spec() -> None:
    arg_spec = ArgSpec(args={'a': int}, kwargs={'b': str}, return_=list[str])
    assert list(arg_spec.args) == [int]
    assert dict(arg_spec.kwargs) == {'b': str}
    assert arg_spec.return_ == list[str]


def combine_numbers(a: int, *, b: float) -> str:
    return f"{a} and {b}"


T = TypeVar('T', int, float)


def complicated_append(a: T, *, b: list[T]) -> tuple[T, ...]:
    b.append(a)

    return tuple(b)


def test_arg_spec_from_function_simple() -> None:
    arg_spec = ArgSpec.from_function(combine_numbers)
    assert list(arg_spec.args) == [int]
    assert dict(arg_spec.kwargs) == {'b': float}


def test_arg_spec_from_function_typevar() -> None:
    arg_spec = ArgSpec.from_function(complicated_append)

    assert list(arg_spec.args) == [T]  # type: ignore[comparison-overlap]
    assert dict(arg_spec.kwargs) == {'b': list[T]}  # type: ignore[valid-type]
    specific_arg_spec = arg_spec.bind_type_vars(bound={T: int})
    assert list(specific_arg_spec.args) == [int]
    assert dict(specific_arg_spec.kwargs) == {'b': list[int]}


def test_arg_spec_decorated_function_with_wraps() -> None:
    from collections.abc import Callable

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        from functools import wraps

        @wraps(func)
        def wrapper(*args: float, **kwargs: float) -> str:
            return "Wrapped: " + func(*args, **kwargs)

        return wrapper

    @decorator
    def decorated(a: int, *, b: float) -> str:
        return f"{a} and {b}"

    arg_spec = ArgSpec.from_function(decorated)
    assert list(arg_spec.args) == [int]
    assert dict(arg_spec.kwargs) == {'b': float}


def test_arg_spec_decorated_function_without_wraps() -> None:
    from collections.abc import Callable

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        def wrapper(*args: float, **kwargs: float) -> str:
            return "Wrapped: " + func(*args, **kwargs)

        return wrapper

    @decorator
    def decorated(a: int, *, b: float) -> str:
        return f"{a} and {b}"

    arg_spec = ArgSpec.from_function(decorated)
    assert list(arg_spec.args) == []
    assert dict(arg_spec.kwargs) == {}
