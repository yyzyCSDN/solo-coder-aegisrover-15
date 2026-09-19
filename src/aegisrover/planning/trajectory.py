"""Path smoothing and time parameterisation with explicit feasibility reports.

Smoothing must keep the first and last waypoint exactly: a smoothed path that starts
at the second waypoint leaves the robot short of its real departure point. Time
parameterisation then assigns a speed profile that starts and ends at rest while
respecting a maximum speed, acceleration and jerk everywhere.

The profile is a jerk-limited S-curve: acceleration ramps between zero and its peak
at +/- max_jerk instead of jumping instantly, which removes the start/stop shock and
the high-frequency torque changes that shake the frame on long runs. When the path is
too short (or the limits too tight) to honour the requested endpoint or cruising
speeds, the planner silently slows the profile instead of declaring it infeasible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = ('TrajectorySample', 'Trajectory', 'smooth', 'time_parameterize', 'polyline_length')

Point = tuple[float, float]

_EPS = 1e-12
_BISECT_ITERS = 70


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
    feasible: bool
    max_jerk: float = math.inf
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
                'max_jerk': None if math.isinf(self.max_jerk) else self.max_jerk,
                'feasible': self.feasible, 'violations': list(self.violations),
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
                      max_jerk: float | None = None,
                      start_speed: float = 0.0, end_speed: float = 0.0) -> Trajectory:
    """Assign a jerk-limited S-curve speed profile along a polyline.

    max_jerk bounds the rate of change of acceleration. With a finite jerk limit the
    acceleration eases in/out instead of stepping, so the robot neither lurches at
    start/stop nor injects high-frequency torque changes while cruising. When None it
    defaults to ``2 * max_accel`` (a roughly one-second ramp through the acceleration
    envelope); pass ``float('inf')`` to reproduce a plain acceleration-limited profile.

    The profile never comes back infeasible for positive limits: if the polyline is
    shorter than the distance needed to honour the requested cruising or endpoint
    speeds, those speeds are automatically lowered until the manoeuvre fits.
    """
    if max_speed <= 0 or max_accel <= 0:
        raise ValueError('max_speed and max_accel must be positive')
    if max_jerk is not None and max_jerk <= 0:
        raise ValueError('max_jerk must be positive or None')
    if start_speed < 0.0 or end_speed < 0.0:
        raise ValueError('endpoint speeds must be non-negative')
    jerk = 2.0 * max_accel if max_jerk is None else float(max_jerk)
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        sample = TrajectorySample(0.0, 0.0, 0.0, pts[0]) if pts else TrajectorySample(0.0, 0.0, 0.0, (0.0, 0.0))
        return Trajectory((sample,), 0.0, max_speed, max_accel, True, jerk)
    distances = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    cumulative = [0.0]
    for distance in distances:
        cumulative.append(cumulative[-1] + distance)
    if cumulative[-1] <= _EPS:
        samples = tuple(TrajectorySample(0.0, 0.0, 0.0, point) for point in pts)
        return Trajectory(samples, 0.0, max_speed, max_accel, True, jerk)

    v0, v1 = min(start_speed, max_speed), min(end_speed, max_speed)
    profile = _plan_profile(cumulative[-1], max_speed, max_accel, jerk, v0, v1)
    # The planner may have lowered the endpoints to fit a short arc: report the speeds
    # the profile actually honours rather than the (now aspirational) requests.
    v0, v1 = profile.v_start, profile.v_end

    def sample_speed(index: int) -> float:
        # Rest endpoints are exact zeros; interior samples invert the phase cubic.
        if index == 0:
            return v0
        if index == len(pts) - 1:
            return v1
        return profile.speed_at(cumulative[index])

    samples = tuple(TrajectorySample(profile.time_at(cumulative[i]), cumulative[i],
                                     sample_speed(i), pts[i])
                    for i in range(len(pts)))
    violations = _violations(profile, max_speed, max_accel, jerk)
    return Trajectory(samples, profile.total_time, max_speed, max_accel, not violations, jerk, violations)


# --------------------------------------------------------------------------- S-curve
class _Phase:
    """One constant-jerk piece of the profile, parametrised by local time tau."""

    __slots__ = ('dt', 'v0', 'a0', 'j')

    def __init__(self, dt: float, v0: float, a0: float, jerk: float):
        self.dt = dt
        self.v0 = v0
        self.a0 = a0
        self.j = jerk

    def speed(self, tau: float) -> float:
        return self.v0 + self.a0 * tau + 0.5 * self.j * tau * tau

    def accel(self, tau: float) -> float:
        return self.a0 + self.j * tau

    def dist_to(self, tau: float) -> float:
        return self.v0 * tau + 0.5 * self.a0 * tau * tau + self.j * tau ** 3 / 6.0


class _Profile:
    """Piecewise constant-jerk scalar speed profile v(s) over an arc of length L."""

    __slots__ = ('length', 'phases', 'bounds', 'total_time', 'v_start', 'v_end')

    def __init__(self, length: float, phases: list[_Phase], v_start: float = 0.0,
                 v_end: float = 0.0):
        self.length = length
        self.phases = phases
        self.v_start = v_start
        self.v_end = v_end
        self.bounds: list[tuple[float, float, float]] = []  # (s_start, s_end, t_start)
        s = t = 0.0
        for phase in phases:
            self.bounds.append((s, s + phase.dist_to(phase.dt), t))
            s += phase.dist_to(phase.dt)
            t += phase.dt
        self.total_time = t

    def _locate(self, target_s: float) -> tuple[_Phase, float, float]:
        chosen = 0
        for index, (s0, s1, _t0) in enumerate(self.bounds):
            if s0 - _EPS <= target_s < s1 - _EPS:
                chosen = index
                break
            if target_s >= s1 - _EPS:
                chosen = index
        s0, s1, t0 = self.bounds[chosen]
        phase = self.phases[chosen]
        if phase.dt <= _EPS:
            return phase, 0.0, t0
        if target_s <= s0 + _EPS:
            return phase, 0.0, t0
        if target_s >= s1 - _EPS:
            return phase, phase.dt, t0
        lo, hi = 0.0, phase.dt
        for _ in range(_BISECT_ITERS):
            mid = (lo + hi) / 2.0
            if phase.dist_to(mid) < target_s - s0:
                lo = mid
            else:
                hi = mid
        return phase, (lo + hi) / 2.0, t0

    def speed_at(self, s: float) -> float:
        phase, tau, _ = self._locate(min(max(s, 0.0), self.length))
        return max(0.0, phase.speed(tau))

    def time_at(self, s: float) -> float:
        _, tau, t0 = self._locate(min(max(s, 0.0), self.length))
        return t0 + tau

    def state_at_time(self, t: float) -> tuple[float, float]:
        """Return (speed, acceleration) at a uniform time-grid instant."""
        t = min(max(t, 0.0), self.total_time)
        t0_acc = 0.0
        for phase in self.phases:
            if t <= t0_acc + phase.dt + _EPS:
                tau = min(max(t - t0_acc, 0.0), phase.dt)
                return max(0.0, phase.speed(tau)), phase.accel(tau)
            t0_acc += phase.dt
        last = self.phases[-1]
        return max(0.0, last.speed(last.dt)), last.accel(last.dt)


def _accel_half(v_start: float, v_end: float, a_max: float, jerk: float) -> list[_Phase]:
    """Phases ramping speed up from v_start to v_end, acceleration leaving/ending at zero."""
    if math.isinf(jerk):
        return [_Phase((v_end - v_start) / a_max, v_start, a_max, 0.0)]
    if v_end <= v_start + _EPS:
        return []
    dv = v_end - v_start
    if dv >= a_max * a_max / jerk:
        # Trapezoidal acceleration pulse: full a_max plateau between the jerk ramps.
        ap = a_max
        tj = ap / jerk
        tv = (dv - ap * ap / jerk) / ap
        first = _Phase(tj, v_start, 0.0, jerk)
        second = _Phase(tv, first.speed(tj), ap, 0.0)
        third_v0 = second.speed(tv)
        return [first, second, _Phase(tj, third_v0, ap, -jerk)]
    ap = math.sqrt(dv * jerk)  # triangular jerk: no constant-acceleration plateau
    tj = ap / jerk
    return [
        _Phase(tj, v_start, 0.0, jerk),
        _Phase(tj, v_start + 0.5 * jerk * tj * tj, ap, -jerk),
    ]


def _mirror_half(phases: list[_Phase], v_start: float, v_end: float) -> list[_Phase]:
    """Reflect an accelerating half (v_start -> v_end) into a braking half (v_end -> v_start).

    Reflecting every velocity about the midpoint (v_start+v_end)/2 and negating
    acceleration and jerk preserves the phase timing while swapping its direction.
    """
    offset = v_start + v_end
    return [_Phase(p.dt, offset - p.v0, -p.a0, -p.j) for p in phases]


def _half_distance(v_start: float, v_end: float, a_max: float, jerk: float) -> float:
    return sum(p.dist_to(p.dt) for p in _accel_half(v_start, v_end, a_max, jerk))


def _minimum_join_distance(va: float, vb: float, a_max: float, jerk: float) -> float:
    """Shortest arc on which an S-curve can leave va and arrive at vb.

    If one endpoint is faster, the quickest admissible join decelerates the higher
    speed straight down to the lower one (an accelerating half run backwards).
    """
    if vb > va + _EPS:
        return _half_distance(va, vb, a_max, jerk)
    if va > vb + _EPS:
        braking = _mirror_half(_accel_half(vb, va, a_max, jerk), vb, va)
        return sum(p.dist_to(p.dt) for p in braking)
    return 0.0


def _join_phases(va: float, vb: float, a_max: float, jerk: float) -> list[_Phase]:
    """S-curve phases taking the profile from va to vb.

    Accelerates (jerk ramps up then down) when va < vb; brakes via a mirrored
    accelerating half when va > vb; returns nothing for a flat join.
    """
    if vb > va + _EPS:
        return _accel_half(va, vb, a_max, jerk)
    if va > vb + _EPS:
        return _mirror_half(_accel_half(vb, va, a_max, jerk), vb, va)
    return []


def _plan_profile(length: float, v_max: float, a_max: float, jerk: float,
                  v0: float, v1: float) -> _Profile:
    # Degrade gracefully: if the endpoints alone cannot be joined within the arc
    # (e.g. arriving fast at a stop only centimetres away), scale both endpoint speeds
    # down together until the manoeuvre fits. Minimum join distance is a continuous,
    # strictly increasing function of the scale factor, so bisect on the factor itself.
    if _minimum_join_distance(v0, v1, a_max, jerk) > length:
        lo, hi = 0.0, 1.0
        for _ in range(_BISECT_ITERS):
            mid = (lo + hi) / 2.0
            if _minimum_join_distance(v0 * mid, v1 * mid, a_max, jerk) > length:
                hi = mid
            else:
                lo = mid
        factor = (lo + hi) / 2.0
        v0 *= factor
        v1 *= factor

    peak_needs = (_minimum_join_distance(v0, v_max, a_max, jerk)
                  + _minimum_join_distance(v1, v_max, a_max, jerk))
    if peak_needs > length:
        # No cruise segment: search a peak at/above both endpoints whose accelerate/
        # decelerate arc fits. The lower bound is the shortest admissible endpoint join.
        lo, hi = max(v0, v1), v_max
        for _ in range(_BISECT_ITERS):
            mid = (lo + hi) / 2.0
            needs = (_minimum_join_distance(v0, mid, a_max, jerk)
                     + _minimum_join_distance(v1, mid, a_max, jerk))
            if needs > length:
                hi = mid
            else:
                lo = mid
        vp = (lo + hi) / 2.0
        cruise_time = 0.0
    else:
        vp = v_max
        cruise_time = max(0.0, (length - peak_needs) / vp)

    phases: list[_Phase] = []
    phases.extend(_join_phases(v0, vp, a_max, jerk))
    if cruise_time > _EPS:
        v_at_peak = phases[-1].speed(phases[-1].dt) if phases else v0
        phases.append(_Phase(cruise_time, v_at_peak, 0.0, 0.0))
    phases.extend(_join_phases(vp, v1, a_max, jerk))
    return _Profile(length, phases, v0, v1)


def _violations(profile: _Profile, max_speed: float, max_accel: float,
                max_jerk: float) -> tuple[str, ...]:
    problems: list[str] = []
    n_steps = 4096
    h = profile.total_time / n_steps
    states = [profile.state_at_time(k * h) for k in range(n_steps + 1)]
    accels = [a for _, a in states]
    for k, (v, a) in enumerate(states):
        if v > max_speed + 1e-6:
            problems.append(f'speed[{k}]={v:.6f}>{max_speed}')
        if abs(a) > max_accel + 1e-5:
            problems.append(f'accel[{k}]={a:.6f}>{max_accel}')
    if not math.isinf(max_jerk):
        for k in range(1, n_steps):
            jerk = (accels[k + 1] - accels[k]) / h
            if abs(jerk) > max_jerk + 1e-4:
                problems.append(f'jerk[{k}]={jerk:.6f}>{max_jerk}')
    return tuple(problems)
