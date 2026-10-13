"""A scripted driver used as the teacher for the PPO warm start.

A Python port of the Stanley-steering driver in the fevo simulator's tests:
steers from the heading error and centre-line offset (both official inputs)
plus a curvature feed-forward, and follows a grip-limited speed profile with
traction-aware throttle. It needs the track geometry the network doesn't
have; the network learns to copy it from the 15 inputs alone.
"""

from __future__ import annotations

import math

import fevo


class Expert:
    def __init__(self, track: fevo.Track, rules: fevo.Rules, grip_margin: float = 0.85) -> None:
        self.track = track
        self.car = rules.car
        t = track.raw
        n = track.points
        self.n = n
        c = t["centerline"]
        x = lambda i: c[2 * (i % n)]  # noqa: E731
        y = lambda i: c[2 * (i % n) + 1]  # noqa: E731
        self.k = []
        for i in range(n):
            h1 = math.atan2(y(i) - y(i - 5), x(i) - x(i - 5))
            h2 = math.atan2(y(i + 5) - y(i), x(i + 5) - x(i))
            d = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
            self.k.append(d / 5)
        car = self.car
        G = car["gravityMps2"]
        self.G = G
        v = []
        for i in range(n):
            kmax = max(abs(self.k[(i + j) % n]) for j in range(-20, 21))
            vv = 30.0
            for _ in range(20):
                grip = grip_margin * car["tyreMuFront"] * (G + 0.5 * car["airDensityKgM3"] * vv * vv * car["downforceAreaM2"] / car["massKg"])
                vv = min(90.0, math.sqrt(grip / max(kmax, 1e-6)))
            v.append(vv)
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                v[i] = min(v[i], math.sqrt(v[(i + 1) % n] ** 2 + 2 * 3.5 * G * t["spacingM"]))
        self.v = v

    def act(self, obs: list[float] | tuple[float, ...], idx: int, vx: float) -> tuple[float, float]:
        car, n = self.car, self.n
        t = self.track.raw
        half = (t["widthLeft"][idx] + t["widthRight"][idx]) / 2
        heading_err = obs[14] * math.pi
        offset_left = -obs[13] * half
        ahead = (idx + round(0.3 * vx / t["spacingM"])) % n
        k_ahead = self.k[ahead]
        delta = heading_err + math.atan(-4 * offset_left / (vx + 2)) + math.atan(car["wheelbaseM"] * k_ahead)
        steer = -delta / car["maxSteerRad"]
        q = 0.5 * car["airDensityKgM3"] * vx * vx
        rear_load = car["massKg"] * self.G * car["cgToFrontM"] / car["wheelbaseM"] + q * car["downforceAreaM2"] * (1 - car["aeroBalanceFront"])
        lat_share = min(1.0, vx * vx * abs(k_ahead) / (car["tyreMuFront"] * (self.G + q * car["downforceAreaM2"] / car["massKg"])))
        spare = math.sqrt(max(0.0, 1 - lat_share * lat_share))
        usable = min(1.0, 0.8 * spare * car["tyreMuRear"] * rear_load * max(vx, 1.0) / car["powerW"])
        # Off the line the exact traction limit is a throttle of about 0.01, too fine for a network
        # to copy; a little wheelspin costs almost nothing and is far easier to learn.
        usable = max(usable, 0.15) if vx < 20 else usable
        throttle = max(-1.0, min(usable, (self.v[ahead] - vx) * 0.4))
        return max(-1.0, min(1.0, steer)), throttle
