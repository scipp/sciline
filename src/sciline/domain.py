# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from types import GenericAlias, NoneType
from typing import (
    TYPE_CHECKING,
    Any,
    ForwardRef,
    Generic,
    TypeVar,
    TypeVarTuple,
    get_args,
    get_origin,
)

PARAM = TypeVar("PARAM")
PARAMS = TypeVarTuple("PARAMS")
SUPER = TypeVar("SUPER")
NATIVE_SCOPE = TypeVar("NATIVE_SCOPE")
_TYPE_VAR_TUPLE = type(PARAMS)


def _check_supertype(cls: type, scope_cls: type) -> None:
    # Mypy does not support __orig_bases__ yet(?)
    # See also https://stackoverflow.com/a/73746554 for useful info
    scope = cls.__orig_bases__[0]  # type: ignore[attr-defined]
    # Only check direct subclasses
    if get_origin(scope) is scope_cls:
        supertype = get_args(scope)[-1]
        # Remove potential generic params
        # In Python 3.8, get_origin does not work with numpy.typing.NDArray,
        # but it defines __origin__
        supertype = getattr(supertype, '__origin__', None) or supertype
        if supertype not in cls.__bases__:
            raise TypeError(
                f"Missing or wrong interface for {cls}, "
                f"should inherit {supertype}.\n"
                "Example:\n"
                "\n"
                "    Param = TypeVar('Param')\n"
                "    \n"
                "    class A(sl.Scope[Param, float], float):\n"
                "        ...\n"
            )


class Scope(Generic[*PARAMS, SUPER]):
    """
    Helper for defining a generic type alias.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _check_supertype(cls, Scope)
        return super().__init_subclass__(**kwargs)

    def __new__(cls, x: SUPER) -> SUPER:  # type: ignore[misc]
        return x


class _NativeScopeBase:
    def __init__(self, args: tuple[Any, ...]) -> None:
        self.args = args

    def __mro_entries__(self, bases: tuple[object, ...]) -> tuple[()]:
        return ()


if TYPE_CHECKING:

    class NativeScope(Generic[*PARAMS, SUPER]):
        """Static typing interface for :func:`native_scope`."""

        def __new__(cls, x: SUPER) -> SUPER: ...  # type: ignore[misc]

else:

    class NativeScope:
        """
        Erased base class for defining a scoped native type.

        Use it with :func:`native_scope` as the first base class:

        .. code-block:: python

            @sciline.native_scope
            class NeXusData(
                sciline.NativeScope[RunType, sc.DataArray], sc.DataArray
            ):
                pass

        The ``NativeScope`` base is erased during class creation, so
        ``NeXusData`` has only ``sc.DataArray`` as a runtime base.
        """

        @classmethod
        def __class_getitem__(cls, args: Any) -> _NativeScopeBase:
            if not isinstance(args, tuple):
                args = (args,)
            return _NativeScopeBase(args)


def _native_scope_getitem(cls: type, args: Any) -> GenericAlias:
    if not isinstance(args, tuple):
        args = (args,)
    args = tuple(
        NoneType if arg is None else ForwardRef(arg) if isinstance(arg, str) else arg
        for arg in args
    )
    parameters = getattr(cls, '__parameters__', ())
    for parameter in parameters:
        if prepare_subst := getattr(parameter, '__typing_prepare_subst__', None):
            args = prepare_subst(cls, args)
    if not parameters:
        raise TypeError(f"{cls} is not a generic class")
    if len(args) != len(parameters):
        kind = 'many' if len(args) > len(parameters) else 'few'
        raise TypeError(
            f"Too {kind} arguments for {cls}; actual {len(args)}, "
            f"expected {len(parameters)}"
        )
    args = tuple(
        item
        for parameter, argument in zip(parameters, args, strict=True)
        for item in (
            argument if isinstance(parameter, _TYPE_VAR_TUPLE) else (argument,)
        )
    )
    return GenericAlias(cls, args)


def _native_scope_new(cls: type, x: Any) -> Any:
    return x


def native_scope(cls: type[NATIVE_SCOPE]) -> type[NATIVE_SCOPE]:
    """
    Turn a :class:`NativeScope` declaration into a generic, identity key.

    Apply this decorator to a class whose first base is ``NativeScope`` and
    whose remaining direct base is its value type:

    .. code-block:: python

        @sciline.native_scope
        class NeXusData(
            sciline.NativeScope[RunType, sc.DataArray], sc.DataArray
        ):
            pass

    Both ``NeXusData[RunType](data)`` and ``NeXusData(data)`` return ``data``
    unchanged. Type checkers retain the generic ``NativeScope`` declaration.

    ``NativeScope`` is erased while the class is defined, allowing the remaining
    base to be a native type that does not support multiple inheritance.
    """
    orig_bases = cls.__dict__.get('__orig_bases__', ())
    scope = orig_bases[0] if orig_bases else None
    if not isinstance(scope, _NativeScopeBase):
        raise TypeError(
            f"Missing NativeScope declaration for {cls}.\n"
            "Example:\n"
            "\n"
            "    @sl.native_scope\n"
            "    class A(sl.NativeScope[Param, float], float):\n"
            "        ...\n"
        )
    if not scope.args:
        raise TypeError(f"NativeScope declaration for {cls} has no interface type")

    supertype = scope.args[-1]
    supertype = getattr(supertype, '__origin__', None) or supertype
    if supertype not in cls.__bases__:
        raise TypeError(
            f"Missing or wrong interface for {cls}, should inherit {supertype}.\n"
            "Example:\n"
            "\n"
            "    @sl.native_scope\n"
            "    class A(sl.NativeScope[Param, float], float):\n"
            "        ...\n"
        )

    parameters = '__parameters__'
    setattr(cls, parameters, GenericAlias(cls, scope.args[:-1]).__parameters__)
    class_getitem = '__class_getitem__'
    setattr(cls, class_getitem, classmethod(_native_scope_getitem))
    new = '__new__'
    setattr(cls, new, staticmethod(_native_scope_new))
    return cls
