"""The low-speed blend loophole: does a driver lose speed faster than its tyres and drag allow?

sim-core blends to a kinematic model when the FORWARD speed vx is under dynamicAboveMps (6 m/s), overwriting the
sideways velocity and yaw rate with their kinematic values. A spinning car has little forward speed but can still
move sideways fast, so that blend deletes its sideways speed (and kinetic energy) within a few steps: free braking.
This scans laps for steps where the car's total speed falls faster than every tyre at peak grip plus drag allows,
by driver and circuit, and whether the lap was finished.

    uv run python -m fevoracer.loophole_scan
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import fevo

from . import general, realtracks
from .notebook2 import Notebook
from .paths import WINNERS


def lap(sim: fevo.Simulator, seed: int, nb: Notebook | None, car: dict) -> tuple[str, int, list[tuple[int, float, float, float]]]:
    """Laps stepped one by one (notes applied as notebook2 does); the steps over the limit: (point, total km/h
    before, km/h lost, forward km/h)."""
    c = sim.core.call
    n = sim.track.points
    m, g = car["massKg"], car["gravityMps2"]
    mu = max(car["tyreMuFront"], car["tyreMuRear"])
    obs, st, over = sim.reset(seed), "running", []
    while st == "running":
        j = sim.progress_index % n
        steer, own = sim.evaluate_program(obs)
        own = float(np.clip(own, -1, 1))
        steer, thr = nb.apply(j, float(steer), own, obs) if nb else (float(steer), own)
        vx, vy = c("getVx"), c("getVy")
        st = sim.step(float(np.clip(steer, -1, 1)), float(np.clip(thr, -1, 1)))
        obs = sim.observation()
        if st == "crashed":
            break  # the crash step stops the car by rule; not driving
        v0, v1 = math.hypot(vx, vy), math.hypot(c("getVx"), c("getVy"))
        q = 0.5 * car["airDensityKgM3"] * v0 * v0
        limit = (mu * (m * g + q * car["downforceAreaM2"]) + q * car["dragAreaM2"]) / m + car["rollingResistance"] * g
        if (v0 - v1) * sim.rules.agent_hz > 1.05 * limit:
            over.append((j, v0 * 3.6, (v0 - v1) * 3.6, vx * 3.6))
    return st, sim.steps, over


def main() -> None:
    setup_d = json.loads((WINNERS / "general-solo4-es.json").read_text())["setup"]
    sim = fevo.Simulator(general.RULES)
    setup = [setup_d[k["id"]] for k in sim.rules.setup]
    sim.set_setup(setup)
    car = sim.rules.car_for(setup)
    rows = []

    def record(who: str, spec: str, st: str, steps: int, over: list) -> None:
        rows.append((who, spec, st, steps, over))

    tracks = realtracks.names("train") + realtracks.names("test")
    for name in ("general-solo4-es", "general-solo4-robust2-laps", "general-solo4-mixd-r30", "general-solo4-slow", "general-solo4-avg"):
        sim.load_program(fevo.from_json((WINNERS / f"{name}.json").read_text()))
        for spec in tracks:
            sim.load_track(general._track(spec))
            for s in range(3):
                record(name, spec, *lap(sim, s, None, car))
    for folder in ("notebook", "notebook2"):
        for f in sorted((WINNERS / folder).glob("*.json")):
            r = json.loads(f.read_text())
            if "notebook" not in r:
                continue
            nbj = r["notebook"]
            sim.load_track(general._track(r["track"]))
            sim.load_program(fevo.from_json((WINNERS / f"{r['driver']}.json").read_text()))
            k = len(nbj["points"])
            nb = Notebook(sim.track.points, nbj["points"], nbj.get("apex", nbj["points"]), nbj.get("power", nbj["points"]), 0.2)
            nb.notes["brake"][:] = nbj.get("brake", nbj.get("shift_m", [0.0] * k))
            nb.notes["line"][:] = nbj.get("line", [0.0] * k)
            nb.notes["exit"][:] = nbj.get("exit", [0.0] * k)
            for s in range(3):
                record(f"{folder}", r["track"], *lap(sim, s, nb, car))
    for folder in ("practice", "practice2", "practice3", "practice4", "distill-teachers"):
        d = WINNERS / folder
        if not (d / "report.json").exists():
            continue
        for r in json.loads((d / "report.json").read_text()):
            last = r["points"][-1]["budget"]
            f = d / f"{r['track'].replace('gen:', 'g')}-{last}.json"
            if not f.exists():
                continue
            sim.load_track(general._track(r["track"]))
            sim.load_program(fevo.from_json(f.read_text()))
            for s in range(3):
                record(f"{folder}-{last}", r["track"], *lap(sim, s, None, car))

    hit = [r for r in rows if r[4]]
    print(f"{len(rows)} laps scanned; {len(hit)} use free braking at least once "
          f"({sum(1 for r in hit if r[2] == 'finished')} of them finish the lap)")
    by = {}
    for who, spec, st, steps, over in hit:
        by.setdefault(who, []).append((spec, st, steps, over))
    for who, items in by.items():
        print(f"\n{who}: {len(items)} laps")
        for spec, st, steps, over in items[:12]:
            worst = max(over, key=lambda o: o[2])
            print(f"  {spec:<14} {st:<9} {fevo.format_lap_time(steps) if st == 'finished' else '':>8}  {len(over):2d} steps, worst: "
                  f"{worst[2]:.0f} km/h lost in one step from {worst[1]:.0f} km/h (forward {worst[3]:.0f} km/h) at point {worst[0]}")
    Path("logs").mkdir(exist_ok=True)
    Path("logs/loophole_scan.json").write_text(json.dumps([[w, s, st, n, o] for w, s, st, n, o in rows]) + "\n")


if __name__ == "__main__":
    main()
