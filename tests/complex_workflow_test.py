# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import NewType, TypeVar

import numpy as np
import numpy.typing as npt

import sciline as sl


@dataclass
class RawData:
    data: npt.NDArray[np.float64]
    monitor1: float
    monitor2: float


SampleRun = NewType('SampleRun', int)
BackgroundRun = NewType('BackgroundRun', int)
DetectorMask = NewType('DetectorMask', npt.NDArray[np.int64])
DirectBeam = NewType('DirectBeam', npt.NDArray[np.float64])
SolidAngle = NewType('SolidAngle', npt.NDArray[np.float64])

Run = TypeVar('Run')


# TODO Giving the base twice works with mypy, how can we avoid typing it twice?
class Raw(sl.Scope[Run, RawData], RawData):
    ...


class Masked(sl.Scope[Run, npt.NDArray[np.float64]], npt.NDArray[np.float64]):
    ...


class IncidentMonitor(sl.Scope[Run, float], float):
    ...


class TransmissionMonitor(sl.Scope[Run, float], float):
    ...


class TransmissionFraction(sl.Scope[Run, float], float):
    ...


class IofQ(sl.Scope[Run, npt.NDArray[np.float64]], npt.NDArray[np.float64]):
    ...


BackgroundSubtractedIofQ = NewType('BackgroundSubtractedIofQ', npt.NDArray[np.float64])


def incident_monitor(x: Raw[Run]) -> IncidentMonitor[Run]:
    return IncidentMonitor(x.monitor1)


def transmission_monitor(x: Raw[Run]) -> TransmissionMonitor[Run]:
    return TransmissionMonitor(x.monitor2)


def mask_detector(x: Raw[Run], mask: DetectorMask) -> Masked[Run]:
    return Masked(x.data * mask)


def transmission(
    incident: IncidentMonitor[Run], transmission: TransmissionMonitor[Run]
) -> TransmissionFraction[Run]:
    return TransmissionFraction(incident / transmission)


def iofq(
    x: Masked[Run],
    solid_angle: SolidAngle,
    direct_beam: DirectBeam,
    transmission: TransmissionFraction[Run],
) -> IofQ[Run]:
    return IofQ(x / (solid_angle * direct_beam * transmission))


reduction = [incident_monitor, transmission_monitor, mask_detector, transmission, iofq]
params = {
    Raw[SampleRun]: Raw(RawData(data=np.ones(4), monitor1=1.0, monitor2=2.0)),
    Raw[BackgroundRun]: Raw(RawData(data=np.ones(4) * 1.5, monitor1=1.0, monitor2=4.0)),
    DetectorMask: DetectorMask(np.array([1, 1, 0, 1])),
    SolidAngle: SolidAngle(np.array([1.0, 0.5, 0.25, 0.125])),
    DirectBeam: DirectBeam(np.array(1 / 1.5)),
}


def subtract_background(
    sample: IofQ[SampleRun], background: IofQ[BackgroundRun]
) -> BackgroundSubtractedIofQ:
    return BackgroundSubtractedIofQ(sample - background)


def test_reduction_workflow() -> None:
    # See https://github.com/python/mypy/issues/14661
    pipeline = sl.Pipeline(
        [subtract_background] + reduction,  # type: ignore[operator]
        params=params,
    )

    assert np.array_equal(pipeline.compute(IofQ[SampleRun]), [3, 6, 0, 24])
    assert np.array_equal(pipeline.compute(IofQ[BackgroundRun]), [9, 18, 0, 72])
    assert np.array_equal(pipeline.compute(BackgroundSubtractedIofQ), [-6, -12, 0, -48])
