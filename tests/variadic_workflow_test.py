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

from sciline.graph import find_nodes_in_paths
from sciline.task_graph import TaskGraph
from sciline.variadic import Map

T = TypeVar('T')


@dataclass(frozen=True)
class Key:
    label: type
    tp: type
    index: int


def _make_mapping_provider(values, Value, tp, Index):
    def provider(*args: Value) -> tp:
        return tp(dict(zip(values[Index], args)))

    return provider


def _make_instance_provider(value):
    return lambda: value


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
            Label, Value = get_args(tp)
            size = len(indices[Label])
            provider = _make_mapping_provider(indices, Value, tp, Label)
            args = [Key(Label, Value, i) for i in range(size)]
            graph[tp] = (provider, args)

            subgraph = build(providers, indices, Value)
            path = find_nodes_in_paths(subgraph, Value, Label)
            for key, value in subgraph.items():
                if key in path:
                    for i in range(size):
                        provider, args = value
                        args = [
                            Key(Label, arg, i) if arg in path else arg for arg in args
                        ]
                        graph[Key(Label, key, i)] = (provider, args)
                else:
                    graph[key] = value
            for i, label in enumerate(indices[Label]):
                graph[Key(Label, Label, i)] = (_make_instance_provider(label), ())
        elif (provider := providers.get(tp)) is not None:
            args = get_type_hints(provider)
            del args['return']
            graph[tp] = (provider, tuple(args.values()))
            for arg in args.values():
                if arg not in graph:
                    stack.append(arg)
        else:
            raise RuntimeError(f'No provider for {tp}')
    return graph


def get(
    providers: Dict[type, Callable[..., Any]],
    indices: Dict[type, Iterable],
    tp: Type[T],
) -> TaskGraph:
    graph = build(providers, indices, tp)
    return TaskGraph(graph=graph, keys=tp)


def test_Map():
    Filename = NewType('Filename', str)
    Config = NewType('Config', int)
    Image = NewType('Image', float)
    CleanedImage = NewType('CleanedImage', float)
    ScaledImage = NewType('ScaledImage', float)
    Param = NewType('Param', float)
    ImageParam = NewType('ImageParam', float)
    Result = NewType('Result', float)

    def read(filename: Filename) -> Image:
        return Image(float(filename[-1]))

    def image_param(filename: Filename) -> ImageParam:
        return ImageParam(sum(ord(c) for c in filename))

    def clean2(x: Image, param: ImageParam) -> CleanedImage:
        return x * param

    def clean(x: Image) -> CleanedImage:
        return x

    def scale(x: CleanedImage, param: Param, config: Config) -> ScaledImage:
        return x * param + config

    def combine_old(
        images: Map[Filename, ScaledImage], params: Map[Filename, ImageParam]
    ) -> float:
        return sum(images.values())

    def combine(images: Map[Filename, ScaledImage]) -> float:
        return sum(images.values())

    def combine_configs(x: Map[Config, float]) -> Result:
        return Result(sum(x.values()))

    def make_int() -> int:
        return 2

    def make_param() -> Param:
        return 2.0

    filenames = tuple(f'file{i}' for i in range(6))
    configs = tuple(range(2))
    indices = {Filename: filenames, Config: configs}
    providers = {
        Image: read,
        CleanedImage: clean,
        ScaledImage: scale,
        float: combine,
        int: make_int,
        Param: make_param,
        ImageParam: image_param,
        Result: combine_configs,
    }

    graph = get(providers, indices, int)
    assert graph.compute() == 2
    graph = get(providers, indices, Result)
    assert graph.compute() == 66.0
    # graph.visualize().render('graph', format='png')
    from dask.delayed import Delayed

    Delayed(Result, graph._graph).visualize(filename='graph.png')
