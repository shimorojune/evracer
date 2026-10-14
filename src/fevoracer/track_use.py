"""How much of the track a driver uses: time with tyres over the edge lines and how far past them it goes.

Tyre points are the simulator's (sim-core allTyresOff): the outer edge of each tyre, at the axles
(cgToFront ahead of the centre, wheelbase - cgToFront behind) and the car's full width. A tyre is over
the line when that point is past an edge; under solo-3 the lap only ends once all four are.

    uv run python -m fevoracer.track_use winners/solo3-spielberg.json --rules solo-3
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import fevo


def lap_use(sim: fevo.Simulator, seed: int, car: dict) -> dict:
    t = sim.track.raw
    c = np.array(t["centerline"]).reshape(-1, 2)
    n = len(c)
    tan = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    nl = np.stack([-tan[:, 1], tan[:, 0]], 1)
    wl, wr = np.array(t["widthLeft"]), np.array(t["widthRight"])
    front, rear, hw = car["cgToFrontM"], car["cgToFrontM"] - car["wheelbaseM"], car["widthM"] / 2
    lap = sim.run_lap(seed, trace=True)
    tr = lap.trace
    over_any = over2 = 0
    max_tyre = max_centre = -math.inf
    idx = 0
    for x, y, yaw in zip(tr["x"], tr["y"], tr["yaw"]):
        cy, sy = math.cos(yaw), math.sin(yaw)
        past = []
        for fx, fy in ((front, hw), (front, -hw), (rear, hw), (rear, -hw)):
            px, py = x + fx * cy - fy * sy, y + fx * sy + fy * cy
            cand = np.arange(idx - 30, idx + 31) % n
            i = int(cand[np.argmin(np.sum((c[cand] - (px, py)) ** 2, axis=1))])
            off = float(np.dot((px - c[i][0], py - c[i][1]), nl[i]))
            past.append(max(off - wl[i], -off - wr[i]))  # metres beyond the nearer edge (negative: inside)
        cand = np.arange(idx - 30, idx + 31) % n
        idx = int(cand[np.argmin(np.sum((c[cand] - (x, y)) ** 2, axis=1))])
        off = float(np.dot((x - c[idx][0], y - c[idx][1]), nl[idx]))
        max_centre = max(max_centre, off - wl[idx], -off - wr[idx])
        k = sum(p > 0 for p in past)
        over_any += k >= 1
        over2 += k >= 2
        max_tyre = max(max_tyre, max(past))
    hz = sim.rules.agent_hz
    return {"seed": seed, "status": lap.status, "lap": fevo.format_lap_time(lap.steps) if lap.finished else lap.status,
            "tyre_over_s": over_any / hz, "two_tyres_over_s": over2 / hz, "share": over_any / len(tr["x"]),
            "max_tyre_past_m": max_tyre, "max_centre_past_m": max_centre}


def main() -> None:
    ap = argparse.ArgumentParser(prog="track-use", description="Time with tyres over the edge lines, and the furthest excursion")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("--rules", default="solo-3")
    a = ap.parse_args()
    for f in a.files:
        mlp = json.load(open(f))
        sim = fevo.Simulator(a.rules)
        sim.load_track(a.track)
        sim.load_program(fevo.from_json(json.dumps(mlp)))
        car = sim.rules.car_for(sim.setup)
        print(f"{f} ({a.rules})")
        for s in range(len(sim.rules.seeds)):
            u = lap_use(sim, s, car)
            print(f"  seed {s}: {u['lap']:>9}  a tyre over the line {u['tyre_over_s']:5.2f} s ({100 * u['share']:4.1f}% of the lap), "
                  f"two or more {u['two_tyres_over_s']:5.2f} s, furthest tyre {u['max_tyre_past_m']:+.2f} m past the line, "
                  f"car centre {u['max_centre_past_m']:+.2f} m (negative = inside)")


if __name__ == "__main__":
    main()
