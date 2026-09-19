"""Legacy thin wrapper kept for import compatibility.

The jerk-aware implementation lives in :mod:`aegisrover.planning.trajectory`. It
returns (time, speed) pairs just like the old acceleration-only helper, but the
underlying profile eases acceleration in/out and slows itself down automatically
when the limits or path length would otherwise be infeasible.
"""
from __future__ import annotations

import math

from .trajectory import time_parameterize


def parameterize(points, max_speed, max_accel, max_jerk=None):
    trajectory = time_parameterize(points, max_speed=max_speed, max_accel=max_accel,
                                   max_jerk=max_jerk)
    return [(sample.t, sample.v) for sample in trajectory.samples]
