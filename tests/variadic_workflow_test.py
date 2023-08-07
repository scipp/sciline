# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline as sl


def test_Map() -> None:
    Run = NewType('Run', int)
    Setting = NewType('Setting', int)
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

    def clean(x: Image) -> CleanedImage:
        return CleanedImage(x)

    def scale(x: CleanedImage, param: Param, config: Config) -> ScaledImage:
        return ScaledImage(x * param + config)

    def combine(images: sl.Series[Run, ScaledImage]) -> float:
        return sum(images.values())

    def combine_configs(x: sl.Series[Setting, float]) -> Result:
        return Result(sum(x.values()))

    def make_int() -> int:
        return 2

    def make_param() -> Param:
        return Param(2.0)

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
    pipeline.set_param_table(sl.ParamTable(Run, {Filename: filenames}))
    pipeline.set_param_table(sl.ParamTable(Setting, {Config: configs}))

    graph = pipeline.get(int)
    assert graph.compute() == 2
    graph = pipeline.get(Result)
    assert graph.compute() == 66.0
    # graph.visualize().render('graph', format='png')
    # from dask.delayed import Delayed
    # dsk = {key: (value, *args) for key, (value, args) in graph._graph.items()}
    # Delayed(Result, dsk).visualize(filename='graph.png')
