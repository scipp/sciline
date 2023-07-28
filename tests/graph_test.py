# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

import sciline as sl
from sciline.graph import find_path, find_unique_path


def test_find_path():
    graph = {"D": ["B", "C"], "C": ["A"], "B": ["A"]}
    assert find_path(graph, "D", "A") == ["D", "B", "A"]


def test_find_unique_path():
    graph = {"D": ["B", "C"], "C": ["A"], "B": ["A"]}
    with pytest.raises(RuntimeError):
        find_unique_path(graph, "D", "A")
    graph = {"D": ["B", "C"], "C": ["A"], "B": ["aux"]}
    assert find_unique_path(graph, "D", "A") == ["D", "C", "A"]


# 1. starting from join point, search dependencies recursively, until
#    we find the fork point.
# 2. use find_unique_path to find the path from fork point to join point.
# 3. replace all nodes inside path by nodes mapped of size of fork
# 4. insert special fork and join nodes

# Do we need to be more explicit in what we combine?
#     def combine(images: Multi[Image]) -> float:
#         return sum(images)
# ... does not specify the "dimension" we reduce over.
#
# filenames = Structured('subrun', ['x.dat', 'y.dat'])

from typing import Literal, Generic, TypeVar, List, Iterator
from collections.abc import Collection

T = TypeVar('T')
DIM = TypeVar('DIM')


class Group(Collection[T], Generic[DIM, T]):
    def __init__(self, value: List[T]) -> None:
        self._stack = value

    def __contains__(self, item: object) -> bool:
        return item in self._stack

    def __iter__(self) -> Iterator[T]:
        return iter(self._stack)

    def __len__(self) -> int:
        return len(self._stack)


def test_literal_param() -> None:
    X = Literal['x']

    def fork() -> Group[X, int]:
        return Group[X, int]([1, 2, 3])

    def process(a: int) -> float:
        return 0.5 * a

    def join(a: Group[X, float]) -> float:
        return sum(a)

    assert join(Group([1.0, 2.0, 3.0])) == 6.0


def test_literal_comp():
    assert Literal['x'] == Literal['x']


from typing import get_type_hints


def wrap(func):
    arg_types = get_type_hints(func)

    def wrapper(x: List[arg_types['x']]) -> List[arg_types['return']]:
        return [func(x_) for x_ in x]

    return wrapper


def test_decorator():
    """
    Given a funcion
        f(a: A, b: B, c: C, ...) -> Ret: ...
    and tps=(B, Ret) return a new function
        f(a: A, b: List[B], c: C, ...) -> List[Ret]: ...
    """

    def f(x: int, y: str) -> float:
        return 0.5 * x

    assert wrap(f)([1, 2, 3]) == [0.5, 1.0, 1.5]
    g = wrap(f)
    assert get_type_hints(g)['x'] == List[int]
    assert get_type_hints(g)['return'] == List[float]


def test_pipeline():
    from typing import NewType

    Filename = NewType('Filename', str)
    Data = NewType('Data', float)
    Param = NewType('Param', str)
    Run = TypeVar('Run')
    Raw = NewType('Raw', float)
    Clean = NewType('Clean', float)
    SampleRun = NewType('SampleRun', int)

    pl = sl.Pipeline()
    filenames = ['x.dat', 'y.dat']
    params = ['a', 'b']
    pl.set_mapping_keys(Filename, filenames)
    # pl.indices[Filename] = filenames ??
    # pl[Filename, 'x.dat']  # returns new pipeline, restricted to single filename (or range)

    def clean(raw: Raw[Run]) -> Clean[Run]:
        return Clean(raw.data)

    def combine(data: sl.Mapping[Filename, Clean[SampleRun]]) -> float:
        return sum(data.values())

    # pipeline has Filename index, so when building graph and looking for
    # sl.Mapping[Filename, Data] it will know how many tasks to create.
