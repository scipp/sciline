# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import NewType, TypeVar

import dask
import numpy as np

import sciline as sl

# We use dask with a single thread, to ensure that call counting below is correct.
dask.config.set(scheduler='synchronous')


@dataclass
class RawData:
    data: np.ndarray
    monitor1: float
    monitor2: float


SampleRun = NewType('SampleRun', int)
BackgroundRun = NewType('BackgroundRun', int)
DetectorMask = NewType('DetectorMask', np.ndarray)
DirectBeam = NewType('DirectBeam', np.ndarray)
SolidAngle = NewType('SolidAngle', np.ndarray)

Run = TypeVar('Run')


class Raw(sl.Scope[Run], RawData):
    ...


class Masked(sl.Scope[Run], np.ndarray):
    ...


class IncidentMonitor(sl.Scope[Run], float):
    ...


class TransmissionMonitor(sl.Scope[Run], float):
    ...


class TransmissionFraction(sl.Scope[Run], float):
    ...


class IofQ(sl.Scope[Run], np.ndarray):
    ...


BackgroundSubtractedIofQ = NewType('BackgroundSubtractedIofQ', np.ndarray)


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


def raw_sample() -> Raw[SampleRun]:
    return Raw[SampleRun](RawData(data=np.ones(4), monitor1=1.0, monitor2=2.0))


def raw_background() -> Raw[BackgroundRun]:
    return Raw[BackgroundRun](
        RawData(data=np.ones(4) * 1.5, monitor1=1.0, monitor2=4.0)
    )


def detector_mask() -> DetectorMask:
    return DetectorMask(np.array([1, 1, 0, 1]))


def solid_angle() -> SolidAngle:
    return SolidAngle(np.array([1.0, 0.5, 0.25, 0.125]))


def direct_beam() -> DirectBeam:
    return DirectBeam(np.array(1 / 1.5))


def subtract_background(
    sample: IofQ[SampleRun], background: IofQ[BackgroundRun]
) -> BackgroundSubtractedIofQ:
    return BackgroundSubtractedIofQ(sample - background)


def test_reduction_workflow():
    container = sl.Container(
        [
            raw_sample,
            raw_background,
            detector_mask,
            solid_angle,
            direct_beam,
            subtract_background,
        ]
        + reduction
    )

    assert np.array_equal(container.get(IofQ[SampleRun]), [3, 6, 0, 24])
    assert np.array_equal(container.get(IofQ[BackgroundRun]), [9, 18, 0, 72])
    assert np.array_equal(container.get(BackgroundSubtractedIofQ), [-6, -12, 0, -48])
