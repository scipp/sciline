# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sys
from typing import NewType, TypeVar, TypeVarTuple, get_args, get_origin

import pytest

import sciline as sl

T = TypeVar("T")


class _NativeBase:
    pass


def test_mypy_detects_wrong_arg_type_of_Scope_subclass() -> None:
    Param = TypeVar('Param')
    Param1 = NewType('Param1', int)

    class A(sl.Scope[Param, float], float): ...

    A[Param1](1.5)
    A[Param1]('abc')  # type: ignore[arg-type]


def test_mypy_detects_wrong_arg_type_of_NativeScope_subclass() -> None:
    Param = TypeVar('Param')
    Param1 = NewType('Param1', int)

    @sl.native_scope
    class A(sl.NativeScope[Param, float], float): ...

    A[Param1](1.5)
    A[Param1]('abc')  # type: ignore[arg-type]


def test_missing_interface_of_scope_subclass_raises() -> None:
    param = TypeVar('param')

    with pytest.raises(TypeError, match="Missing or wrong interface for"):

        class A(sl.Scope[param, float]): ...


def test_mypy_accepts_interface_of_scope_sibling_class() -> None:
    param = TypeVar('param')
    param1 = NewType('param1', int)

    class A(sl.Scope[param, float], float): ...

    a = A[param1](1.5)
    a + a


def test_Scope_inconsistent_type_and_interface_raises() -> None:
    param = TypeVar('param')

    with pytest.raises(TypeError, match="Missing or wrong interface for"):

        class A(sl.Scope[param, str], float): ...


def test_Scope_two_params_inconsistent_type_and_interface_raises() -> None:
    Param1 = TypeVar('Param1')
    Param2 = TypeVar('Param2')

    with pytest.raises(TypeError, match="Missing or wrong interface for"):

        class A(sl.Scope[Param1, Param2, str], float): ...


def test_Scope_two_params() -> None:
    Param1 = TypeVar('Param1')
    Param2 = TypeVar('Param2')

    class A(sl.Scope[Param1, Param2, float], float): ...

    assert isinstance(A[int, int](1.5), float)


def test_native_scope_alias_preserves_generic_key_and_value() -> None:
    Param = TypeVar('Param')

    @sl.native_scope
    class A(sl.NativeScope[Param, _NativeBase], _NativeBase):
        pass

    value = _NativeBase()
    key = A[int]
    assert A.__bases__ == (_NativeBase,)
    assert get_origin(key) is A
    assert get_args(key) == (int,)
    assert get_args(A[None]) == (type(None),)
    assert get_args(A['int'])[0].__forward_arg__ == 'int'
    assert key(value) is value
    assert A(x=value) is value


def test_native_scope_rejects_wrong_number_of_type_arguments() -> None:
    Param = TypeVar('Param')

    @sl.native_scope
    class A(sl.NativeScope[Param, _NativeBase], _NativeBase):
        pass

    with pytest.raises(TypeError, match="Too few arguments"):
        A[()]  # type: ignore[misc]
    with pytest.raises(TypeError, match="Too many arguments"):
        A[int, str]  # type: ignore[misc]

    @sl.native_scope
    class B(sl.NativeScope[_NativeBase], _NativeBase):
        pass

    with pytest.raises(TypeError, match="is not a generic class"):
        B[()]  # type: ignore[misc]


def test_native_scope_supports_variadic_scope_parameters() -> None:
    Params = TypeVarTuple('Params')

    @sl.native_scope
    class A(sl.NativeScope[*Params, _NativeBase], _NativeBase):
        pass

    assert get_args(A[()]) == ()
    assert get_args(A[int, str]) == (int, str)


def test_native_scope_inconsistent_type_and_interface_raises() -> None:
    Param = TypeVar('Param')

    with pytest.raises(TypeError, match="Missing or wrong interface for"):

        @sl.native_scope
        class A(sl.NativeScope[Param, str], _NativeBase):
            pass


def test_native_scope_keys_work_in_pipeline() -> None:
    Sample = NewType('Sample', int)
    Background = NewType('Background', int)
    Run = TypeVar('Run', Sample, Background)
    Detector = NewType('Detector', int)
    Monitor = NewType('Monitor', int)
    Component = TypeVar('Component', Detector, Monitor)

    @sl.native_scope
    class Raw(sl.NativeScope[Component, Run, _NativeBase], _NativeBase):
        pass

    @sl.native_scope
    class Processed(sl.NativeScope[Component, Run, _NativeBase], _NativeBase):
        pass

    def process(data: Raw[Component, Run]) -> Processed[Component, Run]:
        return Processed[Component, Run](data)

    value = _NativeBase()
    pipeline = sl.Pipeline([process], params={Raw[Detector, Sample]: value})
    assert pipeline.compute(Processed[Detector, Sample]) is value


def test_native_scope_generic_key_is_automatically_specialized() -> None:
    Param = TypeVar('Param', int, float)

    @sl.native_scope
    class A(sl.NativeScope[Param, _NativeBase], _NativeBase):
        pass

    value = _NativeBase()
    pipeline = sl.Pipeline(params={A: value})
    assert pipeline.compute(A[int]) is value
    assert pipeline.compute(A[float]) is value


if sys.version_info >= (3, 13):

    def test_native_scope_applies_default_type_arguments() -> None:
        Param = TypeVar('Param', default=int)

        @sl.native_scope
        class A(sl.NativeScope[Param, _NativeBase], _NativeBase):
            pass

        assert get_args(A[()]) == (int,)
