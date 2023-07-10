# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Dict, NewType


def parametrized_domain_type(name: str, base: type) -> type:
    """
    Return a type-factory for parametrized domain types.

    The types return by the factory are created using typing.NewType. The returned
    factory is used similarly to a Generic, but note that the factory itself should
    not be used for annotations.

    Parameters
    ----------
    name:
        The name of the type. This is used as a prefix for the names of the types
        returned by the factory.
    base:
        The base type of the types returned by the factory.
    """

    class Factory:
        _subtypes: Dict[str, type] = {}

        def __class_getitem__(cls, tp: type) -> type:
            key = f'{name}_{tp.__name__}'
            if (t := cls._subtypes.get(key)) is None:
                t = NewType(key, base)
                cls._subtypes[key] = t
            return t

    return Factory
