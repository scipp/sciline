# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NewType,
    Tuple,
    Type,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import dask

from sciline.graph import find_all_paths
from sciline.variadic import Map

T = TypeVar('T')


@dataclass(frozen=True)
class Key:
    key: type
    tp: type
    index: int


def _make_mapping_provider(values, Value, tp, Index):
    def provider(*args: Value) -> tp:
        return tp(dict(zip(values[Index], args)))

    return provider


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
            size = len(indices[Index])
            provider = _make_mapping_provider(indices, Value, tp, Index)
            args = [Key(Index, Value, i) for i in range(size)]
            graph[tp] = (provider, *args)

            subgraph = build(providers, indices, Value)
            paths = find_all_paths(subgraph, Value, Index)
            path = set()
            for p in paths:
                path.update(p)
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
    ImageParam = NewType('ImageParam', float)

    def read(filename: Filename) -> Image:
        return Image(float(filename[-1]))

    def image_param(filename: Filename) -> ImageParam:
        return ImageParam(sum(ord(c) for c in filename))

    def clean2(x: Image, param: ImageParam) -> CleanedImage:
        return x * param

    def clean(x: Image) -> CleanedImage:
        return x

    def scale(x: CleanedImage, param: Param) -> ScaledImage:
        return x * param

    def combine(
        images: Map[Filename, ScaledImage], params: Map[Filename, ImageParam]
    ) -> float:
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
        ImageParam: image_param,
    }

    graph = build(providers, indices, int)
    assert dask.get(graph, int) == 2
    graph = build(providers, indices, float)
    from dask.delayed import Delayed

    Delayed(float, graph).visualize(filename='graph.png')
    assert dask.get(graph, float) == 30.0
