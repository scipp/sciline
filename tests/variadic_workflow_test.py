# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline as sl


def test_Map() -> None:
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
        images: sl.Map[Filename, ScaledImage], params: sl.Map[Filename, ImageParam]
    ) -> float:
        return sum(images.values())

    def combine(images: sl.Map[Filename, ScaledImage]) -> float:
        return sum(images.values())

    def combine_configs(x: sl.Map[Config, float]) -> Result:
        return Result(sum(x.values()))

    def make_int() -> int:
        return 2

    def make_param() -> Param:
        return 2.0

    filenames = tuple(f'file{i}' for i in range(6))
    configs = tuple(range(2))
    pipeline = sl.Pipeline(
        [
            read,
            clean,
            scale,
            combine,
            combine_configs,
            make_int,
            make_param,
            image_param,
        ]
    )
    pipeline.set_index(Filename, filenames)
    pipeline.set_index(Config, configs)

    graph = pipeline.get(int)
    assert graph.compute() == 2
    graph = pipeline.get(Result)
    assert graph.compute() == 66.0
    graph.visualize().render('graph', format='png')
    # from dask.delayed import Delayed
    # dsk = {key: (value, *args) for key, (value, args) in graph._graph.items()}
    # Delayed(Result, dsk).visualize(filename='graph.png')
