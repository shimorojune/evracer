"""Where does the time go? A reference lap and a gate-by-gate comparison.

The reference is a quasi-steady-state point-mass lap along the centre line,
using the rules' car numbers: cornering limited by tyre grip plus downforce,
acceleration by power and rear traction, braking by tyres and brakes, with a
friction circle in between. It ignores yaw dynamics and uses the centre line
rather than a racing line, so it's a guide, not a bound: a car that uses the
track width can beat it in places.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import fevo


@dataclass
class Reference:
    speed: list[float]  # m/s per centre-line point
    time_at: list[float]  # seconds from the start line to each point
    lap_s: float


def _curvature(track: fevo.Track, half_window: int = 6) -> list[float]:
    c = track.raw["centerline"]
    n = track.points
    x = lambda i: c[2 * (i % n)]  # noqa: E731
    y = lambda i: c[2 * (i % n) + 1]  # noqa: E731
    k = []
    for i in range(n):
        h1 = math.atan2(y(i) - y(i - half_window), x(i) - x(i - half_window))
        h2 = math.atan2(y(i + half_window) - y(i), x(i + half_window) - x(i))
        d = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
        k.append(abs(d) / (half_window * track.raw["spacingM"]))
    return k


def reference_lap(track: fevo.Track, rules: fevo.Rules | None = None) -> Reference:
    car = (rules or fevo.load_rules()).car
    m, g, rho = car["massKg"], car["gravityMps2"], car["airDensityKgM3"]
    L, a = car["wheelbaseM"], car["cgToFrontM"]
    b = L - a
    muF, muR = car["tyreMuFront"], car["tyreMuRear"]
    bal = car["aeroBalanceFront"]
    n = track.points
    ds = track.raw["spacingM"]
    k = _curvature(track)

    def loads(v: float) -> tuple[float, float]:
        df = 0.5 * rho * car["downforceAreaM2"] * v * v
        return m * g * b / L + df * bal, m * g * a / L + df * (1 - bal)

    def lat_cap(v: float) -> float:
        fzf, fzr = loads(v)
        return muF * fzf + muR * fzr  # N

    # Cornering speed limit per point (fixed-point solve, since grip grows with v²).
    vmax = []
    for ki in k:
        v = 90.0
        for _ in range(30):
            v = min(100.0, math.sqrt(lat_cap(v) / m / max(ki, 1e-6)))
        vmax.append(v)

    def spare(v: float, ki: float) -> float:
        need = m * v * v * ki
        cap = lat_cap(v)
        return math.sqrt(max(0.0, 1 - (need / cap) ** 2)) if cap > 0 else 0.0

    drag = lambda v: 0.5 * rho * car["dragAreaM2"] * v * v + car["rollingResistance"] * m * g  # noqa: E731

    # Forward: accelerate from a standstill, power- and traction-limited.
    v = [0.0] * (n + 1)
    for i in range(n):
        vi = v[i]
        _, fzr = loads(vi)
        drive = min(car["powerW"] / max(vi, 1.0), muR * fzr * spare(vi, k[i]))
        acc = (drive - drag(vi)) / m
        v[i + 1] = min(vmax[(i + 1) % n], math.sqrt(max(0.0, vi * vi + 2 * acc * ds)))
    # Backward: brake into every corner (twice round, so the end of the lap brakes for turn 1 too).
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            vn = v[i + 1]
            fzf, fzr = loads(vn)
            brake = min(car["maxBrakeForceN"], muF * fzf + muR * fzr) * spare(vn, k[(i + 1) % n])
            dec = (brake + drag(vn)) / m
            v[i] = min(v[i], math.sqrt(vn * vn + 2 * dec * ds)) if i > 0 else 0.0
    t = [0.0]
    for i in range(n):
        avg = max(0.5, 0.5 * (v[i] + v[i + 1]))
        t.append(t[-1] + ds / avg)
    return Reference(v[:n], t[:n], t[n])


def gate_losses(track: fevo.Track, trace: dict[str, list[float]], ref: Reference, hz: int = 60) -> list[dict[str, float]]:
    """Time per gate-to-gate section: the agent's lap against the reference."""
    gates = track.gates + [track.points]
    first_step: dict[int, int] = {}
    for step, g in enumerate(trace["gates"]):
        first_step.setdefault(g, step)
    rows = []
    for s in range(len(gates) - 1):
        if s not in first_step or s + 1 not in first_step and s + 1 < len(gates) - 1:
            break
        t0 = first_step[s] / hz
        t1 = (first_step.get(s + 1, len(trace["gates"]) - 1)) / hz
        r0 = ref.time_at[gates[s]]
        r1 = ref.time_at[gates[s + 1]] if s + 1 < len(gates) - 1 else ref.lap_s
        a, b = gates[s], gates[s + 1]
        seg = trace["speed"][first_step[s] : first_step.get(s + 1, len(trace["speed"]))]
        rows.append({
            "section": s, "from_m": a * track.raw["spacingM"], "agent_s": t1 - t0, "ref_s": r1 - r0,
            "loss_s": (t1 - t0) - (r1 - r0), "agent_min_kmh": min(seg) * 3.6 if seg else 0.0,
            "ref_min_kmh": min(ref.speed[a:b]) * 3.6, "agent_max_kmh": max(seg) * 3.6 if seg else 0.0,
            "ref_max_kmh": max(ref.speed[a:b]) * 3.6,
        })  # fmt: skip
    return rows
