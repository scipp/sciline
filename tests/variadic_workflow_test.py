# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import (
    NewType,
    TypeVar,
    Generic,
    Iterable,
    Tuple,
    Dict,
    Callable,
    get_type_hints,
    get_origin,
    get_args,
    List,
    Type,
    Any,
)


import sciline as sl
from sciline.variadic import Stack, Map
from sciline.graph import find_path, find_unique_path


T = TypeVar('T')


class Multi(Generic[T]):
    def __init__(self, values: Iterable[T]) -> None:
        self.values = values


def test_literal_param() -> None:
    Filename = NewType('Filename', str)
    Image = NewType('Image', float)

    def read_file(filename: Filename) -> Image:
        print(filename)
        return Image(float(filename[-1]))

    def combine(images: Multi[Image]) -> float:
        return sum(images.values)

    filenames = [f'file{i}' for i in range(10)]
    # pipeline = sl.Pipeline([read_file, combine])
    # pipeline[Multi[Filename]] = Multi(filenames)
    # assert pipeline.compute(float) == 45.0

    # -> Multi[Image] not found
    # -> look for Image provider
    # -> look for Filename provider
    # -> look for Multi[Filename] provider

    @dataclass(frozen=True)
    class Key:
        tp: type
        index: int

    # How do write down graph for this?
    # - need dummy __getitem__ tasks
    # - need dummy to_list tasks
    graph = {}
    graph[Multi[Filename]] = Multi(filenames)
    size = len(filenames)
    for i in range(size):
        graph[Key(Filename, i)] = (lambda x, j: x.values[j], Multi[Filename], i)
        graph[Key(Image, i)] = (read_file, Key(Filename, i))
    graph[Multi[Image]] = (
        lambda *args: Multi(args),
        *[Key(Image, i) for i in range(size)],
    )
    graph[float] = (combine, Multi[Image])

    import dask

    assert dask.get(graph, Multi[Filename]).values == filenames
    assert dask.get(graph, float) == 45.0


from inspect import getfullargspec, signature, Parameter
from typing import get_type_hints


def test_args_hints() -> None:
    def f(*myargs: int) -> None:
        pass

    args_name = getfullargspec(f)[1]
    params = signature(f).parameters
    for p in params.values():
        if p.kind == Parameter.VAR_POSITIONAL:
            args_name = p.name
            args_type = p.annotation
            break
    assert args_name == 'myargs'
    assert args_type == int
    # args_name = signature(f).parameters['args'].name

    assert get_type_hints(f)[args_name] == int


def test_Stack_dependency_uses_Stack_provider():
    Filename = NewType('Filename', str)

    def combine(names: Stack[Filename]) -> str:
        return ';'.join(names)

    filenames = [f'file{i}' for i in range(4)]
    pipeline = sl.Pipeline([combine], params={Stack[Filename]: Stack(filenames)})
    assert pipeline.compute(str) == ';'.join(filenames)


def test_Stack_dependency_maps_provider_over_Stack_provider():
    Filename = NewType('Filename', str)
    Image = NewType('Image', float)

    def read(filename: Filename) -> Image:
        return Image(float(filename[-1]))

    def combine(images: Stack[Image]) -> float:
        return sum(images)

    filenames = tuple(f'file{i}' for i in range(10))
    pipeline = sl.Pipeline([read, combine], params={Stack[Filename]: Stack(filenames)})
    assert pipeline.compute(float) == 45.0


@dataclass(frozen=True)
class Key:
    key: type
    tp: type
    index: int


def build(
    providers: Dict[type, Callable[..., Any]],
    indices: Dict[type, Iterable],
    tp: Type[T],
) -> Dict[type, Tuple[Callable[..., Any], ...]]:
    graph = {}
    stack: List[Type[T]] = [tp]
    while stack:
        tp = stack.pop()
        if tp in indices:
            pass
        elif get_origin(tp) == Map:
            Index, Value = get_args(tp)

            def provider(*values: Value) -> tp:
                return tp(dict(zip(indices[Index], values)))

            size = len(indices[Index])
            args = [Key(Index, Value, i) for i in range(size)]
            graph[tp] = (provider, *args)

            subgraph = build(providers, indices, Value)
            path = find_unique_path(subgraph, Value, Index)
            for key, value in subgraph.items():
                if key in path:
                    for i in range(size):
                        provider, *args = value
                        args = [
                            Key(Index, arg, i) if arg in path else arg for arg in args
                        ]
                        graph[Key(Index, key, i)] = (provider, *args)
                else:
                    graph[key] = value
            for i, index in enumerate(indices[Index]):
                graph[Key(Index, Index, i)] = index
        elif (provider := providers.get(tp)) is not None:
            args = get_type_hints(provider)
            del args['return']
            graph[tp] = (provider, *args.values())
            for arg in args.values():
                if arg not in graph:
                    stack.append(arg)
        else:
            raise RuntimeError(f'No provider for {tp}')
    return graph


def test_Map():
    Filename = NewType('Filename', str)
    Image = NewType('Image', float)
    CleanedImage = NewType('CleanedImage', float)
    ScaledImage = NewType('ScaledImage', float)
    Param = NewType('Param', float)

    def read(filename: Filename) -> Image:
        return Image(float(filename[-1]))

    def clean(x: Image) -> CleanedImage:
        return x

    def scale(x: CleanedImage, param: Param) -> ScaledImage:
        return x * param

    def combine(images: Map[Filename, ScaledImage]) -> float:
        return sum(images.values())

    def make_int() -> int:
        return 2

    def make_param() -> Param:
        return 2.0

    filenames = tuple(f'file{i}' for i in range(6))
    indices = {Filename: filenames}
    providers = {
        Image: read,
        CleanedImage: clean,
        ScaledImage: scale,
        float: combine,
        int: make_int,
        Param: make_param,
    }

    import dask

    graph = build(providers, indices, int)
    assert dask.get(graph, int) == 2
    graph = build(providers, indices, float)
    from dask.delayed import Delayed

    Delayed(float, graph).visualize(filename='graph.png')
    assert dask.get(graph, float) == 30.0
