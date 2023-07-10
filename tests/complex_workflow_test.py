# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Callable, List, NewType

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
Raw = sl.parametrized_domain_type('Raw', RawData)
Masked = sl.parametrized_domain_type('Masked', np.ndarray)
IncidentMonitor = sl.parametrized_domain_type('IncidentMonitor', float)
TransmissionMonitor = sl.parametrized_domain_type('TransmissionMonitor', float)
TransmissionFraction = sl.parametrized_domain_type('TransmissionFraction', float)
IofQ = sl.parametrized_domain_type('IofQ', np.ndarray)
BackgroundSubtractedIofQ = NewType('BackgroundSubtractedIofQ', np.ndarray)


def reduction_factory(tp: type) -> List[Callable]:
    def incident_monitor(x: Raw[tp]) -> IncidentMonitor[tp]:
        return IncidentMonitor[tp](x.monitor1)

    def transmission_monitor(x: Raw[tp]) -> TransmissionMonitor[tp]:
        return TransmissionMonitor[tp](x.monitor2)

    def mask_detector(x: Raw[tp], mask: DetectorMask) -> Masked[tp]:
        return Masked[tp](x.data * mask)

    def transmission(
        incident: IncidentMonitor[tp], transmission: TransmissionMonitor[tp]
    ) -> TransmissionFraction[tp]:
        return TransmissionFraction[tp](incident / transmission)

    def iofq(
        x: Masked[tp],
        solid_angle: SolidAngle,
        direct_beam: DirectBeam,
        transmission: TransmissionFraction[tp],
    ) -> IofQ[tp]:
        return IofQ[tp](x / (solid_angle * direct_beam * transmission))

    return [incident_monitor, transmission_monitor, mask_detector, transmission, iofq]


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
    container = sl.make_container(
        [
            raw_sample,
            raw_background,
            detector_mask,
            solid_angle,
            direct_beam,
            subtract_background,
        ]
        + reduction_factory(SampleRun)
        + reduction_factory(BackgroundRun)
    )
    assert np.array_equal(container.get(IofQ[SampleRun]), [3, 6, 0, 24])
    assert np.array_equal(container.get(IofQ[BackgroundRun]), [9, 18, 0, 72])
    assert np.array_equal(container.get(BackgroundSubtractedIofQ), [-6, -12, 0, -48])
