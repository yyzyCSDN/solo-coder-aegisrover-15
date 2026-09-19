"""Path smoothing and time parameterisation with explicit feasibility reports.

Smoothing must keep the first and last waypoint exactly: a smoothed path that starts
at the second waypoint leaves the robot short of its real departure point. Time
parameterisation then assigns a speed profile that starts and ends at rest while
respecting a maximum speed and a maximum acceleration everywhere.

Passing ``max_jerk`` replaces the trapezoidal profile (which steps acceleration
instantly) with a jerk-limited S-curve: acceleration ramps up and down gradually,
removing the shocks that shake the chassis at start/stop and at the phase
transitions of a long run. When the limits are too tight to reach the configured
speed or acceleration — a short hop, a low jerk cap — the S-curve automatically
trades peak speed and peak acceleration for feasibility instead of reporting a
violation. Only physically impossible boundary conditions (for example not enough
room to decelerate from ``start_speed`` to ``end_speed``) are flagged infeasible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = ('TrajectorySample', 'Trajectory', 'smooth', 'time_parameterize', 'polyline_length')

Point = tuple[float, float]


@dataclass(frozen=True)
class TrajectorySample:
    t: float
    s: float
    v: float
    point: Point


@dataclass(frozen=True)
class Trajectory:
    samples: tuple[TrajectorySample, ...]
    duration: float
    max_speed: float
    max_accel: float
    max_jerk: float | None
    feasible: bool
    violations: tuple[str, ...] = ()

    def at(self, t: float) -> TrajectorySample:
        if not self.samples:
            raise ValueError('empty trajectory')
        for sample in self.samples:
            if sample.t >= t:
                return sample
        return self.samples[-1]

    def to_dict(self) -> dict:
        return {'duration': self.duration, 'max_speed': self.max_speed, 'max_accel': self.max_accel,
                'max_jerk': self.max_jerk, 'feasible': self.feasible, 'violations': list(self.violations),
                'samples': [{'t': round(s.t, 6), 's': round(s.s, 6), 'v': round(s.v, 6),
                             'point': [round(s.point[0], 6), round(s.point[1], 6)]} for s in self.samples]}


def polyline_length(points: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _catmull_rom(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    t2, t3 = t * t, t * t * t

    def axis(a: float, b: float, c: float, d: float) -> float:
        return 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3)

    return (axis(p0[0], p1[0], p2[0], p3[0]), axis(p0[1], p1[1], p2[1], p3[1]))


def smooth(points: Sequence[Point], *, per_segment: int = 8, preserve_endpoints: bool = True) -> list[Point]:
    """Catmull-Rom smoothing that keeps the original first and last waypoint."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 3 or per_segment < 1:
        return pts
    extended = [pts[0], *pts, pts[-1]]
    out: list[Point] = [pts[0]] if preserve_endpoints else []
    for i in range(1, len(extended) - 2):
        for k in range(per_segment):
            out.append(_catmull_rom(extended[i - 1], extended[i], extended[i + 1], extended[i + 2],
                                    k / per_segment))
    out.append(pts[-1])
    if preserve_endpoints:
        if math.dist(out[0], pts[0]) > 1e-9 or math.dist(out[-1], pts[-1]) > 1e-9:
            raise AssertionError('smoothing moved an endpoint')
    return out


def time_parameterize(points: Sequence[Point], *, max_speed: float, max_accel: float,
                      max_jerk: float | None = None, start_speed: float = 0.0,
                      end_speed: float = 0.0) -> Trajectory:
    """Assign a timestamp and speed to every waypoint.

    With ``max_jerk=None`` the profile is trapezoidal (acceleration may step
    instantly). With a positive ``max_jerk`` the profile is a jerk-limited
    S-curve that slows itself down as far as needed to stay feasible.
    ``start_speed``/``end_speed`` are clamped to [0, max_speed].
    """
    if max_speed <= 0 or max_accel <= 0:
        raise ValueError('max_speed and max_accel must be positive')
    if max_jerk is not None and max_jerk <= 0:
        raise ValueError('max_jerk must be positive')
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        sample = TrajectorySample(0.0, 0.0, 0.0, pts[0]) if pts else TrajectorySample(0.0, 0.0, 0.0, (0.0, 0.0))
        return Trajectory((sample,), 0.0, max_speed, max_accel, max_jerk, True)
    distances = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    cumulative = [0.0]
    for distance in distances:
        cumulative.append(cumulative[-1] + distance)
    if max_jerk is not None:
        return _scurve_parameterize(pts, cumulative, max_speed, max_accel, max_jerk,
                                    start_speed, end_speed)
    speeds = [max_speed] * len(pts)
    speeds[0] = min(max_speed, start_speed)
    speeds[-1] = min(max_speed, end_speed)
    for i in range(1, len(pts)):
        speeds[i] = min(speeds[i], math.sqrt(max(0.0, speeds[i - 1] ** 2 + 2 * max_accel * distances[i - 1])))
    for i in range(len(pts) - 2, -1, -1):
        speeds[i] = min(speeds[i], math.sqrt(max(0.0, speeds[i + 1] ** 2 + 2 * max_accel * distances[i])))
    times = [0.0]
    for i in range(1, len(pts)):
        average = (speeds[i - 1] + speeds[i]) / 2.0
        if distances[i - 1] <= 1e-12 or average <= 1e-9:
            times.append(times[-1])
        else:
            times.append(times[-1] + distances[i - 1] / average)
    samples = tuple(TrajectorySample(times[i], cumulative[i], speeds[i], pts[i]) for i in range(len(pts)))
    violations = _violations(samples, distances, max_speed, max_accel)
    return Trajectory(samples, times[-1], max_speed, max_accel, None, not violations, violations)


def _violations(samples: Sequence[TrajectorySample], distances: Sequence[float],
                max_speed: float, max_accel: float) -> tuple[str, ...]:
    problems: list[str] = []
    for index, sample in enumerate(samples):
        if sample.v > max_speed + 1e-9:
            problems.append(f'speed[{index}]={sample.v:.6f}>{max_speed}')
        if index and sample.t < samples[index - 1].t - 1e-12:
            problems.append(f'time[{index}] not monotonic')
    for index, distance in enumerate(distances):
        if distance <= 1e-12:
            continue
        a, b = samples[index], samples[index + 1]
        accel = abs(b.v * b.v - a.v * a.v) / (2.0 * distance)
        if accel > max_accel + 1e-6:
            problems.append(f'accel[{index}]={accel:.6f}>{max_accel}')
    return tuple(problems)


# --------------------------------------------------------------------- S-curves
# A jerk-limited profile is a sequence of constant-jerk phases. Ramps are built
# analytically; when the path is too short to reach max_speed the cruise speed is
# found by bisection, so the profile always degrades gracefully instead of
# becoming infeasible. A phase is a (jerk, duration) pair; phases join with zero
# acceleration unless noted.

def _ramp_phases(delta_v: float, max_accel: float, max_jerk: float) -> tuple[tuple[float, float], ...]:
    """Constant-jerk phases changing speed by delta_v, starting and ending at zero accel."""
    mag = abs(delta_v)
    if mag <= 1e-12:
        return ()
    sign = 1.0 if delta_v > 0 else -1.0
    t_jerk = max_accel / max_jerk
    t_flat = mag / max_accel - t_jerk
    if t_flat > 0.0:
        return ((sign * max_jerk, t_jerk), (0.0, t_flat), (-sign * max_jerk, t_jerk))
    # Too little speed change to ever reach max_accel: triangular accel profile.
    t_peak = math.sqrt(mag / max_jerk)
    return ((sign * max_jerk, t_peak), (-sign * max_jerk, t_peak))


def _phases_distance(phases: Sequence[tuple[float, float]], v_start: float) -> float:
    """Distance covered integrating phases that start at speed v_start and zero accel."""
    v, a, distance = v_start, 0.0, 0.0
    for jerk, duration in phases:
        distance += v * duration + 0.5 * a * duration ** 2 + jerk * duration ** 3 / 6.0
        v += a * duration + 0.5 * jerk * duration ** 2
        a += jerk * duration
    return distance


def _scurve_phases(distance: float, v0: float, vf: float, max_speed: float,
                   max_accel: float, max_jerk: float) -> tuple[tuple[float, float], ...]:
    def span(peak: float) -> float:
        return _phases_distance(_ramp_phases(peak - v0, max_accel, max_jerk)
                                + _ramp_phases(vf - peak, max_accel, max_jerk), v0)

    full = span(max_speed)
    if full <= distance:
        cruise = max(0.0, (distance - full) / max_speed)
        return (_ramp_phases(max_speed - v0, max_accel, max_jerk) + ((0.0, cruise),)
                + _ramp_phases(vf - max_speed, max_accel, max_jerk))
    floor = max(v0, vf)
    if span(floor) > distance:
        # Not enough room to even change speed from v0 to vf within the limits:
        # return the minimal profile and let the feasibility report flag it.
        return (_ramp_phases(floor - v0, max_accel, max_jerk)
                + _ramp_phases(vf - floor, max_accel, max_jerk))
    lo, hi = floor, max_speed
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if span(mid) > distance:
            hi = mid
        else:
            lo = mid
    peak = 0.5 * (lo + hi)
    return (_ramp_phases(peak - v0, max_accel, max_jerk)
            + _ramp_phases(vf - peak, max_accel, max_jerk))


def _integrate_phases(phases: Sequence[tuple[float, float]], v0: float) -> list[tuple[float, float, float, float]]:
    """(t, s, v, a) at every phase boundary, starting from rest accel at (0, 0, v0)."""
    states = [(0.0, 0.0, v0, 0.0)]
    t, s, v, a = 0.0, 0.0, v0, 0.0
    for jerk, duration in phases:
        s += v * duration + 0.5 * a * duration ** 2 + jerk * duration ** 3 / 6.0
        v += a * duration + 0.5 * jerk * duration ** 2
        a += jerk * duration
        t += duration
        states.append((t, s, v, a))
    return states


def _profile_state_at(phases: Sequence[tuple[float, float]], states: Sequence[tuple[float, float, float, float]],
                      s_query: float, hint: int) -> tuple[float, float, int]:
    """(t, v) at arc position s_query, walking forward from phase index hint."""
    k = hint
    while k < len(phases) - 1 and states[k + 1][1] < s_query:
        k += 1
    t0, s0, v0, a0 = states[k]
    jerk, duration = phases[k]
    ds = min(max(s_query - s0, 0.0), states[k + 1][1] - s0)
    if abs(jerk) < 1e-15 and abs(a0) < 1e-15:
        tau = ds / v0 if v0 > 1e-15 else 0.0
    else:
        lo, hi = 0.0, duration
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            covered = v0 * mid + 0.5 * a0 * mid * mid + jerk * mid ** 3 / 6.0
            if covered < ds:
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
    return t0 + tau, v0 + a0 * tau + 0.5 * jerk * tau * tau, k


def _scurve_parameterize(pts: list[Point], cumulative: list[float], max_speed: float, max_accel: float,
                         max_jerk: float, start_speed: float, end_speed: float) -> Trajectory:
    v0 = min(max(start_speed, 0.0), max_speed)
    vf = min(max(end_speed, 0.0), max_speed)
    total = cumulative[-1]
    if total <= 1e-12:
        feasible = abs(v0 - vf) <= 1e-9
        violations = () if feasible else (f'cannot change speed {v0:.6f}->{vf:.6f} over zero distance',)
        samples = tuple(TrajectorySample(0.0, 0.0, v0, point) for point in pts)
        return Trajectory(samples, 0.0, max_speed, max_accel, max_jerk, feasible, violations)
    phases = tuple((jerk, duration) for jerk, duration in
                   _scurve_phases(total, v0, vf, max_speed, max_accel, max_jerk)
                   if duration > 0.0)
    states = _integrate_phases(phases, v0)
    samples = []
    hint = 0
    for point, s_query in zip(pts, cumulative):
        t, v, hint = _profile_state_at(phases, states, min(max(s_query, 0.0), states[-1][1]), hint)
        samples.append(TrajectorySample(t, s_query, max(0.0, v), point))
    # Boundary speeds are exact by construction; undo the bisection round-off so a
    # profile that ends at rest reports exactly 0.0 at the final waypoint.
    first = samples[0]
    samples[0] = TrajectorySample(first.t, first.s, v0, first.point)
    if states[-1][1] <= total + 1e-9:
        last = samples[-1]
        samples[-1] = TrajectorySample(last.t, last.s, vf, last.point)
    samples = tuple(samples)
    violations = _scurve_violations(samples, states, phases, max_speed, max_accel, max_jerk, v0, vf, total)
    return Trajectory(samples, samples[-1].t, max_speed, max_accel, max_jerk, not violations, violations)


def _scurve_violations(samples: tuple[TrajectorySample, ...], states: Sequence[tuple[float, float, float, float]],
                       phases: Sequence[tuple[float, float]], max_speed: float, max_accel: float,
                       max_jerk: float, v0: float, vf: float, total: float) -> tuple[str, ...]:
    problems: list[str] = []
    if states[-1][1] > total + 1e-9:
        problems.append(f'profile needs {states[-1][1]:.6f}m to satisfy speed limits but path is {total:.6f}m')
    if abs(states[0][2] - v0) > 1e-9:
        problems.append(f'start speed {states[0][2]:.6f}!={v0:.6f}')
    if abs(states[-1][2] - vf) > 1e-9:
        problems.append(f'end speed {states[-1][2]:.6f}!={vf:.6f}')
    for _, _, v, a in states:
        if v > max_speed + 1e-9:
            problems.append(f'speed {v:.6f}>{max_speed}')
        if abs(a) > max_accel + 1e-9:
            problems.append(f'accel {a:.6f}>{max_accel}')
    for jerk, _ in phases:
        if abs(jerk) > max_jerk + 1e-9:
            problems.append(f'jerk {jerk:.6f}>{max_jerk}')
    problems.extend(_sample_dynamic_violations(samples, max_speed, max_accel, max_jerk))
    return tuple(problems)


def _sample_dynamic_violations(samples: tuple[TrajectorySample, ...], max_speed: float,
                               max_accel: float, max_jerk: float) -> list[str]:
    """Finite-difference check of the returned samples: average accel per interval is
    exact, and the jerk between interval midpoints is bounded by the profile jerk."""
    problems: list[str] = []
    accels: list[tuple[float, float]] = []
    for index, (a, b) in enumerate(zip(samples, samples[1:]), start=1):
        if b.t < a.t - 1e-12:
            problems.append(f'time[{index}] not monotonic')
        dt = b.t - a.t
        if dt <= 1e-12:
            continue
        if b.v > max_speed + 1e-6:
            problems.append(f'speed[{index}]={b.v:.6f}>{max_speed}')
        accel = (b.v - a.v) / dt
        if abs(accel) > max_accel + 1e-6:
            problems.append(f'accel[{index}]={accel:.6f}>{max_accel}')
        accels.append((0.5 * (a.t + b.t), accel))
    for (t0, a0), (t1, a1) in zip(accels, accels[1:]):
        jerk = (a1 - a0) / (t1 - t0)
        if abs(jerk) > max_jerk + 1e-6:
            problems.append(f'jerk={jerk:.6f}>{max_jerk}')
    return problems
