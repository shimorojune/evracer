"""A fast scripted driver to imitate: the racing line at (nearly) the car's limit.

The old expert (expert.py) follows the centre line at 85% of the front grip.
This one drives the reference racing line (fevo.reference) at a share of the
achievable speed profile from fevoracer.gap: the same car, setup and
per-axle limits, so the target is a speed the car can actually carry.

It uses privileged state (pose, velocities, the track and the line); a
network imitating it only needs its actions.

* Steering: pure pursuit on the line from the rear axle, lookahead growing
  with speed, plus yaw-rate damping.
* Speed: the achievable profile with a margin on cornering speeds only
  (96%, and 90% in slow corners, where steering lock and front slip build-up
  limit the car), so straights stay flat out and braking stays at the limit.
* Throttle: brake when the hardest deceleration any target ahead needs,
  (v² - v_target²) / 2d, reaches 90% of what the tyres can give (late and
  hard, like braking for a marker); otherwise track the target a short way
  ahead. Drive is capped by what the rear tyres have left after cornering
  (their friction circle, from the measured lateral acceleration and load),
  and the expert lifts while the rears slide (solo-2's power slide is caught
  by closing the throttle). Braking is capped per axle with the brake bias and
  load transfer, so the fronts don't lock.

On solo-2 at Spielberg with the solo2safe setup it laps 1:11.3-1:12.0 from
all three starts. It is fragile under sustained steering or throttle errors
(about 0.01 rad or 3% held for half a second), mostly on the exits of T2 and
T3 and through T4.
"""

from __future__ import annotations

import math

import numpy as np

import fevo

from .gap import qss, ref_line


class LineExpert:
    def __init__(self, track: fevo.Track, rules: fevo.Rules, setup: list[float] | None, margin: float = 0.96,
                 grip_use: float = 0.95, look_s: float = 0.30, look_min: float = 6.0, speed_look_s: float = 0.12,
                 kp: float = 3.0, yaw_damp: float = 0.02, line_scale: float = 0.9, k_cross: float = 2.0,
                 brake_on: float = 0.9, drive_use: float = 0.9, lift_on_slide: bool = True, slow_margin: float = 0.9,
                 slow_below: float = 30.0) -> None:
        self.car = car = rules.car_for(setup)
        # The reference line keeps 1.45 m from the edges (0.5 m outside the car's half width): pull it in a little.
        _, c, self.nl, off, _ = ref_line(track)
        self.p = c + self.nl * (off * line_scale)[:, None]
        self.k_cross, self.brake_on, self.drive_use, self.lift_on_slide = k_cross, brake_on, drive_use, lift_on_slide
        self.n = len(self.p)
        # The margin goes on cornering speeds only: straights stay flat out and braking stays at the limit.
        # Slow corners (below slow_below m/s) get a bigger one: there the car is limited by steering lock and
        # how fast the fronts build slip angle, which the quasi-steady limits don't see.
        vmax = qss(self.p, car, vehicle=True, standing=False, h_curv=10).vmax
        scale = np.where(vmax < slow_below, slow_margin, margin)
        self.v = qss(self.p, car, vehicle=True, standing=False, h_curv=10, corner_scale=scale).v[: self.n]
        seg = np.linalg.norm(np.roll(self.p, -1, 0) - self.p, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])  # arc length to each line point (n + 1 values)
        self.brake_look = 320
        self.grip_use, self.look_s, self.look_min, self.speed_look_s = grip_use, look_s, look_min, speed_look_s
        self.kp, self.yaw_damp = kp, yaw_damp
        self.hz = rules.agent_hz
        self.j = 0
        self.vx_prev = 0.0

    def reset(self) -> None:
        self.j = 0
        self.vx_prev = 0.0

    def _nearest(self, x: float, y: float, idx: int) -> int:
        lo = idx - 30
        cand = np.arange(lo, idx + 60) % self.n
        d = np.sum((self.p[cand] - (x, y)) ** 2, axis=1)
        return int(cand[int(np.argmin(d))])

    def __call__(self, sim: fevo.Simulator, obs) -> tuple[float, float]:
        c = sim.core.call
        car = self.car
        x, y, yaw, vx, vy, r = c("getX"), c("getY"), c("getYaw"), c("getVx"), c("getVy"), c("getYawRate")
        idx = c("getProgressIndex")
        if c("getSteps") == 0:
            self.reset()
        j = self._nearest(x, y, idx)
        m, g, L, a = car["massKg"], car["gravityMps2"], car["wheelbaseM"], car["cgToFrontM"]
        b, h = L - a, car["cgHeightM"]

        # Steering: pure pursuit from the rear axle.
        ld = max(self.look_min, self.look_s * vx)
        tx, ty = self.p[(j + int(round(ld))) % self.n]
        rx, ry = x - b * math.cos(yaw), y - b * math.sin(yaw)
        dx, dy = tx - rx, ty - ry
        alpha = math.atan2(dy, dx) - yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        dist = max(1.0, math.hypot(dx, dy))
        delta_left = math.atan2(2 * L * math.sin(alpha), dist) - self.yaw_damp * (r - vx * 2 * math.sin(alpha) / dist)
        # Cross-track correction (Stanley-style) on the error from the line at the front axle.
        fx, fy = x + a * math.cos(yaw), y + a * math.sin(yaw)
        jf = self._nearest(fx, fy, j)
        e_left = float(np.dot((fx - self.p[jf][0], fy - self.p[jf][1]), self.nl[jf]))
        delta_left -= math.atan(self.k_cross * e_left / (vx + 5.0))
        steer = max(-1.0, min(1.0, -delta_left / car["maxSteerRad"]))

        # Loads and what the tyres have left after cornering.
        ax = (vx - self.vx_prev) * self.hz
        self.vx_prev = vx
        q = 0.5 * car["airDensityKgM3"] * vx * vx
        df = q * car["downforceAreaM2"]
        tr = m * ax * h / L
        fzf = max(0.0, m * g * b / L + df * car["aeroBalanceFront"] - tr)
        fzr = max(0.0, m * g * a / L + df * (1 - car["aeroBalanceFront"]) + tr)
        ay = vx * r
        capf, capr = self.grip_use * car["tyreMuFront"] * fzf, self.grip_use * car["tyreMuRear"] * fzr
        fyf, fyr = abs(m * ay * b / L), abs(m * ay * a / L)
        long_f = math.sqrt(max(0.0, capf * capf - fyf * fyf))
        long_r = math.sqrt(max(0.0, capr * capr - fyr * fyr))

        # Brake for the hardest deceleration any target ahead asks for, (v² - v_target²) / 2d, as a driver
        # brakes for a marker; otherwise track the speed target a little ahead, with feed-forward on drag.
        drag = q * car["dragAreaM2"] + car["rollingResistance"] * m * g
        k = np.arange(j + 1, j + self.brake_look + 1)
        d = np.maximum(0.5, (self.s[k % self.n] + (k // self.n) * self.s[self.n]) - self.s[j] - self.speed_look_s * vx)
        need_dec = float(np.max((vx * vx - self.v[k % self.n] ** 2) / (2 * d)))
        ahead = (j + int(round(max(3.0, self.speed_look_s * vx)))) % self.n
        v_t = self.v[ahead]
        bias = car["brakeBiasFront"]
        bmax = min(long_f / bias, long_r / (1 - bias), car["maxBrakeForceN"])
        if need_dec * m > self.brake_on * (bmax + drag):
            f_need = -(need_dec * m - drag)
        else:
            f_need = m * self.kp * (v_t - vx) + drag
        if f_need >= 0:
            # Drive keeps its own margin on the rear friction circle, and the rears sliding means lift (solo-2's
            # power slide is caught by closing the throttle, not by easing off).
            capr_d = capr / self.grip_use * self.drive_use
            drive = min(f_need, math.sqrt(max(0.0, capr_d * capr_d - fyr * fyr)))
            if self.lift_on_slide and c("getRearSlide") > 0:
                drive = 0.0
            throttle = min(1.0, drive * max(vx, 1.0) / car["powerW"])
        else:
            throttle = -min(-f_need, bmax) / car["maxBrakeForceN"]
        return steer, throttle
