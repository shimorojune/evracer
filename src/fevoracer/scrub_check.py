"""Does the car get free braking from steering? (Rohit's recording: Spielberg T1, ~160 km/h lost in ~1.3 s with
near-full lock and almost no brake.)

Per step it logs speed, the car's acceleration (from the change of its world velocity), the steering and pedal,
each axle's slide, and the most the car could decelerate: every tyre at its peak grip, mu x load (static load plus
downforce), plus aero drag and rolling resistance. Any step past that limit would be energy from nowhere. The
simulator keeps a friction circle on each axle; the one place it does not conserve momentum is the blend to a
kinematic model below `dynamicAboveMps` (6 m/s), which the log marks.

Then, across the circuits and drivers, it counts the steps with the pedal off the brake (above -0.1), steering past
0.6 of lock and the car slowing by more than 1.5 g: steering scrub.

    uv run python -m fevoracer.scrub_check
"""

from __future__ import annotations

import json
import math

import numpy as np

import fevo

from . import general, realtracks
from .notebook2 import Notebook, corners, run_lap as nb_lap
from .paths import WINNERS


def trace(sim: fevo.Simulator, seed: int, nb: Notebook | None) -> list[dict]:
    """One lap, stepping as notebook2.run_lap does, recording the car each step."""
    c = sim.core.call
    car = sim.car_params if hasattr(sim, "car_params") else None
    n = sim.track.points
    obs, st, rows = sim.reset(seed), "running", []
    while st == "running":
        j = sim.progress_index % n
        steer, own = sim.evaluate_program(obs)
        own = float(np.clip(own, -1, 1))
        steer, thr = nb.apply(j, float(steer), own, obs) if nb else (float(steer), own)
        vx, vy, yaw = c("getVx"), c("getVy"), c("getYaw")
        st = sim.step(float(np.clip(steer, -1, 1)), float(np.clip(thr, -1, 1)))
        obs = sim.observation()
        rows.append({"j": j, "vx": vx, "vy": vy, "yaw": yaw, "steer": float(np.clip(steer, -1, 1)), "thr": float(np.clip(thr, -1, 1)),
                     "fs": c("getFrontSlide"), "rs": c("getRearSlide")})
    rows.append({"j": sim.progress_index % n, "vx": c("getVx"), "vy": c("getVy"), "yaw": c("getYaw"), "steer": 0, "thr": 0, "fs": 0, "rs": 0,
                 "status": st})
    return rows


def accel(rows: list[dict], hz: int, car: dict) -> list[dict]:
    """World-frame acceleration per step against the grip-plus-drag limit."""
    m, g = car["massKg"], car["gravityMps2"]
    mu = max(car["tyreMuFront"], car["tyreMuRear"])
    out = []
    for a, b in zip(rows, rows[1:]):
        va = np.array([a["vx"] * math.cos(a["yaw"]) - a["vy"] * math.sin(a["yaw"]), a["vx"] * math.sin(a["yaw"]) + a["vy"] * math.cos(a["yaw"])])
        vb = np.array([b["vx"] * math.cos(b["yaw"]) - b["vy"] * math.sin(b["yaw"]), b["vx"] * math.sin(b["yaw"]) + b["vy"] * math.cos(b["yaw"])])
        acc = (vb - va) * hz
        sp = np.linalg.norm(va)
        heading = va / sp if sp > 1e-6 else np.zeros(2)
        v = max(a["vx"], b["vx"])
        q = 0.5 * car["airDensityKgM3"] * v * v
        limit = (mu * (m * g + q * car["downforceAreaM2"]) + q * car["dragAreaM2"] + car["rollingResistance"] * m * g) / m
        out.append({**a, "speed": sp, "a": float(np.linalg.norm(acc)), "along": float(np.dot(acc, heading)), "limit": limit,
                    "blend": min(a["vx"], b["vx"]) < car["dynamicAboveMps"]})
    return out


def main() -> None:
    setup_d = json.loads((WINNERS / "general-solo4-es.json").read_text())["setup"]
    sim = fevo.Simulator(general.RULES)
    setup = [setup_d[k["id"]] for k in sim.rules.setup]
    sim.set_setup(setup)
    car = sim.rules.car_for(setup)
    hz = sim.rules.agent_hz

    # 1. Rohit's session state at Spielberg (notes as the page listed them), start 2.
    sim.load_track("spielberg")
    nbj = json.loads((WINNERS / "notebook2" / "spielberg.json").read_text())["notebook"]
    sim.load_program(fevo.from_json((WINNERS / "general-solo4-mixd-r30.json").read_text()))
    nb = Notebook(sim.track.points, nbj["points"], nbj["apex"], nbj["power"], edge_keep=0.2)
    want = {0.35: ("brake", -11.0), 1.24: ("brake", -18.0), 2.09: ("brake", -15.0), 2.62: ("line", -0.02)}
    for km, (kind, val) in want.items():
        i = int(np.argmin([abs(p / 1000 - km) for p in nb.points]))
        nb.notes[kind][i] = val
    rows = accel(trace(sim, 2, nb), hz, car)
    t1 = [r for r in rows if 200 <= r["j"] <= 520]
    print("Spielberg T1, notes as in the recording, start 2 (every 8th step):")
    print("  point  speed km/h  steer  pedal   decel along path  total accel  limit (grip+drag)  front/rear slide  blend")
    for r in t1[::8]:
        print(f"  {r['j']:5d}  {r['speed'] * 3.6:9.0f}  {r['steer']:+.2f}  {r['thr']:+.2f}  {-r['along'] / 9.81:9.2f} g  {r['a'] / 9.81:9.2f} g"
              f"  {r['limit'] / 9.81:9.2f} g  {r['fs']:.2f}/{r['rs']:.2f}  {'yes' if r['blend'] else ''}")
    worst = max(rows, key=lambda r: r["a"] / r["limit"])
    over = [r for r in rows if r["a"] > 1.02 * r["limit"]]
    print(f"  whole lap: highest total acceleration {worst['a'] / worst['limit']:.0%} of the limit "
          f"(at point {worst['j']}, {worst['speed'] * 3.6:.0f} km/h); steps over the limit: {len(over)}"
          + (f", all in the low-speed blend: {all(r['blend'] for r in over)}" if over else ""))
    lost = max(t1, key=lambda r: r["speed"])["speed"] - min(t1, key=lambda r: r["speed"])["speed"]
    print(f"  T1: {lost * 3.6:.0f} km/h lost, slowest {min(r['speed'] for r in t1) * 3.6:.0f} km/h")

    # 2. How common is steering scrub? General drivers on every circuit, plus the saved notebooks.
    print("\nSteering scrub (pedal above -0.1, steer past 0.6 lock, slowing more than 1.5 g), official start 1:")
    total, hit, over_all = 0, 0, 0
    lines = []
    for name in ("general-solo4-es", "general-solo4-robust2-laps", "general-solo4-mixd-r30"):
        sim.load_program(fevo.from_json((WINNERS / f"{name}.json").read_text()))
        for spec in realtracks.names("train") + realtracks.names("test"):
            sim.load_track(general._track(spec))
            rs = accel(trace(sim, 1, None), hz, car)
            scrub = [r for r in rs if r["thr"] > -0.1 and abs(r["steer"]) > 0.6 and -r["along"] > 1.5 * 9.81]
            total += len(rs)
            hit += len(scrub)
            over_all += sum(r["a"] > 1.02 * r["limit"] and not r["blend"] for r in rs)
            if scrub:
                lines.append(f"  {name:<28} {spec:<14} {len(scrub):3d} steps, worst {max(-r['along'] for r in scrub) / 9.81:.1f} g at "
                             f"{max(scrub, key=lambda r: -r['along'])['speed'] * 3.6:.0f} km/h")
    print("\n".join(lines) or "  none")
    print(f"  {hit} scrub steps out of {total:,} ({hit / total:.2%}); steps over the grip+drag limit outside the low-speed blend: {over_all}")


if __name__ == "__main__":
    main()
