"""Does the expert still lap? Real circuits, generated and slow-corner circuits, on all three official starts.

Compares the racing line with and without fevoracer.line's minimum-radius limit (min_radius 0 is the old line):
circuits lapped on all three starts and the median lap time on each.

    uv run python -m fevoracer.expert_check --gen 40 --slow 40
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

import fevo

from . import general, realtracks
from .general import EXPERT
from .line import racing_line
from .paths import WINNERS
from .racer import LineExpert


def one(args: tuple) -> tuple[str, list]:
    spec, radii = args
    d = json.loads((WINNERS / "solo3-spielberg.json").read_text())["setup"]
    sim = fevo.Simulator(general.RULES)
    setup = [d[k["id"]] for k in sim.rules.setup]
    sim.set_setup(setup)
    tr = general._track(spec)
    sim.load_track(tr)
    out = []
    for r in radii:
        ex = LineExpert(tr, sim.rules, setup, offsets=racing_line(tr, -0.45, min_radius=r), **EXPERT)
        laps = []
        for s in range(3):
            obs, st = sim.reset(s), "running"
            ex.reset()
            while st == "running":
                a = ex(sim, obs)
                st = sim.step(a[0], a[1])
                obs = sim.observation()
            laps.append(sim.steps if st == "finished" else None)
        out.append(laps)
    return spec, out


def main() -> None:
    ap = argparse.ArgumentParser(prog="expert_check", description=__doc__.split("\n")[0])
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--slow", type=int, default=40)
    ap.add_argument("--radii", default="0,12", help="min_radius values to compare (0 is the original line)")
    ap.add_argument("-w", "--workers", type=int)
    a = ap.parse_args()
    radii = [float(r) for r in a.radii.split(",")]
    specs = (realtracks.names("train") + realtracks.names("test")
             + [f"gen:{s}" for s in general._distinct_gen(100, a.gen)] + [f"slow:{s}" for s in range(100, 100 + a.slow)])
    with mp.get_context("spawn").Pool(a.workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(one, [(s, radii) for s in specs])
    groups = {"real": [r for r in res if ":" not in r[0]], "gen": [r for r in res if r[0].startswith("gen:")],
              "slow": [r for r in res if r[0].startswith("slow:")]}
    for name, rows in groups.items():
        line = [f"{name:<5} ({len(rows)} circuits)"]
        for k, r in enumerate(radii):
            ok = [x for _, o in rows for x in [o[k]] if all(v is not None for v in x)]
            line.append(f"min radius {r:g}: {len(ok)}/{len(rows)} lapped")
        both = [o for _, o in rows if all(all(v is not None for v in o[k]) for k in range(len(radii)))]
        if len(radii) == 2 and both:
            dt = sorted((sorted(o[1])[1] - sorted(o[0])[1]) / 60 for o in both)
            line.append(f"median lap change {dt[len(dt) // 2]:+.2f} s (range {dt[0]:+.2f} to {dt[-1]:+.2f})")
        print("  ".join(line), flush=True)
    for spec, o in res:
        if ":" not in spec:
            print(f"  {spec:<14} " + "  ".join(
                f"r{r:g}: " + " / ".join(fevo.format_lap_time(v) if v else "fails" for v in o[k]) for k, r in enumerate(radii)))


if __name__ == "__main__":
    main()
