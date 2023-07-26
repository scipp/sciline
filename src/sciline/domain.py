# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any, Generic, TypeVar, get_args, get_origin

PARAM = TypeVar("PARAM")
SUPER = TypeVar("SUPER")


class Scope(Generic[PARAM, SUPER]):
    """
    Helper for defining a generic type alias.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Mypy does not support __orig_bases__ yet(?)
        # See also https://stackoverflow.com/a/73746554 for useful info
        scope = cls.__orig_bases__[0]  # type: ignore[attr-defined]
        # Only check direct subclasses
        if get_origin(scope) is Scope:
            supertype = get_args(scope)[1]
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
        return super().__init_subclass__(**kwargs)

    def __new__(cls, x: SUPER) -> SUPER:  # type: ignore[misc]
        return x
