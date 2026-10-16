"""Practice on an unseen circuit: how fast a general driver adapts with its own driving alone.

For each circuit the starting driver is whichever general driver goes furthest there (laps finished on the three
official starts first, then completion). It then practises with an evolution strategy on that circuit only: no
expert labels and no racing line, just the driver's own laps and stints scored by progress per second (a crash
adds `crash_s` seconds), with the lap-progress inputs still masked (zero weights, never perturbed).

Each generation a few antithetic pairs of perturbed drivers each drive a lap from an official start and two short
stints from random points of the circuit (random offset, heading and a speed of 25-75 m/s, as explore.py does),
then the mean driver drives the three official starts. Everything driven counts towards the practice budget, in
laps (metres driven / circuit length). The best driver so far (laps finished on the official starts, then
completion, then lap time) is written at each budget checkpoint to winners/practice/<circuit>-<laps>.json.

    uv run python -m fevoracer.practice
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import fevo

from . import general
from .general import EXPERT, MASKED
from .general_es import unflatten
from .line import racing_line
from .paths import WINNERS
from .racer import LineExpert
from .smooth import flatten

from .general import RULES

#: The general drivers practice starts from (the best of them on each circuit): 39-input ones for solo-6's memory.
GENERALS = (["general-solo6-pace2", "general-solo6-r4", "general-solo6-es", "general-solo6"] if RULES == "solo-6"
            else ["general-solo4-robust2-laps", "general-solo4-es", "general-solo4-mixd-r30"])
TRACKS = ["spielberg", "montreal", "monza", "austin", "spa", "gen:900006", "gen:900001"]
CHECKPOINTS = (0, 20, 100, 500)
OUT = WINNERS / "practice"


def _progress(prev: int, now: int, n: int) -> int:
    d = now - prev
    return d + n if d < -n // 2 else d - n if d > n // 2 else d


def official(sim: fevo.Simulator, mlp: dict) -> tuple[list[tuple[str, int, float]], float]:
    """The three official starts: (status, steps, completion) each, and the metres driven."""
    sim.load_program(fevo.from_json(json.dumps(mlp)))
    n, out, metres = sim.track.points, [], 0.0
    for s in range(3):
        sim.reset(s)
        st = sim.run_agent_lap()
        done = 1.0 if st == "finished" else sim.progress_index / n
        out.append((st, sim.steps, done))
        metres += done * n
    return out, metres


def rank(laps: list[tuple[str, int, float]]) -> tuple:
    """Laps finished first, then mean completion, then (faster is better) the median finished lap."""
    fin = sorted(s for st, s, _ in laps if st == "finished")
    return (len(fin), float(np.mean([c for _, _, c in laps])), -(fin[len(fin) // 2] if fin else 10 ** 9))


def drive(sim: fevo.Simulator, mlp: dict, lap_seed: int, starts: list, stint: int, crash_s: float) -> tuple[float, float]:
    """One candidate's practice: a lap from an official start and stints; (fitness, metres driven)."""
    sim.load_program(fevo.from_json(json.dumps(mlp)))
    hz, n = sim.rules.agent_hz, sim.track.points
    rates, metres = [], 0.0
    sim.reset(lap_seed)
    st = sim.run_agent_lap()
    dist = n if st == "finished" else sim.progress_index
    metres += dist
    rates.append(dist / (sim.steps / hz + (0 if st == "finished" else crash_s)))
    for point, lat, hdg, speed in starts:
        sim.reset(1, point=point, lateral_m=lat, heading_rad=hdg, speed=speed)
        prev, dist, st = point, 0, "running"
        for _ in range(stint):
            st = sim.agent_step()
            idx = sim.progress_index
            dist += _progress(prev, idx, n)
            prev = idx
            if st != "running":
                break
        metres += max(0, dist)
        rates.append(max(0, dist) / max(sim.steps / hz + (crash_s if st not in ("running", "finished") else 0), 1e-3))
    return float(np.mean(rates)), metres


def practise(args: tuple) -> dict:
    spec, pairs, sigma, lr, stints, stint, crash_s, budget, seed = args
    setup_d = json.load(open(WINNERS / f"{GENERALS[0]}.json"))["setup"]
    sim = fevo.Simulator(general.RULES)
    sim.set_setup([setup_d[k["id"]] for k in sim.rules.setup])
    tr = general._track(spec)
    sim.load_track(tr)
    n = tr.points
    name = spec.replace("gen:", "g")
    # The expert's lap, only to report how far off the driver is (it never teaches here).
    ex = LineExpert(tr, sim.rules, [setup_d[k["id"]] for k in sim.rules.setup], offsets=racing_line(tr, -0.45), **EXPERT)
    obs, st = sim.reset(1), "running"
    ex.reset()
    while st == "running":
        a = ex(sim, obs)
        st = sim.step(a[0], a[1])
        obs = sim.observation()
    expert = sim.steps if st == "finished" else None
    # Starting driver: the general driver that goes furthest here.
    cands = []
    for g in GENERALS:
        mlp = json.load(open(WINNERS / f"{g}.json"))
        laps, _ = official(sim, mlp)
        cands.append((rank(laps), g, mlp, laps))
    cands.sort(key=lambda c: c[0], reverse=True)
    _, start_name, base, laps0 = cands[0]
    x = flatten(base)
    n_in, n_h = base["layers"][0], base["layers"][1]
    frozen = np.zeros(len(x), bool)
    for j in range(n_h):
        for i in MASKED:
            frozen[j * n_in + i] = True
    rng = np.random.default_rng(seed)
    m, v = np.zeros_like(x), np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_x, best_laps = x.copy(), laps0
    used, gen, t0 = 0.0, 0, time.time()
    report = {"track": spec, "start": start_name, "expert": expert, "points": []}

    def checkpoint(budget_laps: int) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{name}-{budget_laps}.json"
        if not path.exists():
            path.write_text(json.dumps(unflatten(base, best_x)) + "\n")
        report["points"].append({"budget": budget_laps, "used": round(used, 1), "laps": best_laps})

    checkpoint(0)
    marks = [c for c in CHECKPOINTS if c > 0 and c <= budget]
    while marks:
        gen += 1
        starts = [(int(rng.integers(n)), float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-0.05, 0.05)),
                   float(rng.uniform(25, 75))) for _ in range(stints)]
        lap_seed = int(rng.integers(3))
        e = rng.standard_normal((pairs, len(x)))
        e[:, frozen] = 0.0
        fits = []
        for c in [x + sigma * k for k in e] + [x - sigma * k for k in e]:
            f, metres = drive(sim, unflatten(base, c), lap_seed, starts, stint, crash_s)
            fits.append(f)
            used += metres / n
        f = np.array(fits)
        r = np.empty(len(f))
        r[np.argsort(f)] = np.arange(len(f))
        u = r / (len(f) - 1) - 0.5
        g = ((u[:pairs] - u[pairs:])[:, None] * e).sum(0) / (2 * pairs * sigma)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = lr * (m / (1 - b1 ** gen)) / (np.sqrt(v / (1 - b2 ** gen)) + eps)
        step[frozen] = 0.0
        x = x + step
        laps, metres = official(sim, unflatten(base, x))
        used += metres / n
        if rank(laps) > rank(best_laps):
            best_x, best_laps = x.copy(), laps
        while marks and used >= marks[0]:
            checkpoint(marks.pop(0))
    report["generations"], report["minutes"] = gen, round((time.time() - t0) / 60, 1)
    return report


def fmt(laps: list, expert: int | None) -> str:
    fin = sorted(s for st, s, _ in laps if st == "finished")
    comp = float(np.mean([c for _, _, c in laps]))
    if not fin:
        return f"0/3 ({comp:.0%})"
    med = fin[len(fin) // 2]
    gap = f" {100 * (med / expert - 1):+.1f}%" if expert else ""
    return f"{len(fin)}/3 {fevo.format_lap_time(med)}{gap}"


def main() -> None:
    ap = argparse.ArgumentParser(prog="practice", description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", default=",".join(TRACKS))
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=3e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stints", type=int, default=2)
    ap.add_argument("--stint", type=int, default=400)
    ap.add_argument("--crash-s", type=float, default=5.0)
    ap.add_argument("--budget", type=int, default=500, help="laps of practice (all candidates' driving counted)")
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    specs = a.tracks.split(",")
    with mp.get_context("spawn").Pool(a.workers or min(len(specs), max(1, (os.cpu_count() or 2) - 1))) as pool:
        reps = pool.map(practise, [(s, a.pairs, a.sigma, a.lr, a.stints, a.stint, a.crash_s, a.budget, a.seed + i)
                                   for i, s in enumerate(specs)])
    print(f"{'circuit':<12} {'start driver':<28} " + "  ".join(f"{c:>22}" for c in CHECKPOINTS if c <= a.budget) + "  expert")
    for rp in reps:
        cells = [f"{fmt(p['laps'], rp['expert']):>22}" for p in rp["points"]]
        print(f"{rp['track']:<12} {rp['start']:<28} " + "  ".join(cells)
              + f"  {fevo.format_lap_time(rp['expert']) if rp['expert'] else 'fails'}"
              + f"   ({rp['generations']} generations, {rp['minutes']} min)", flush=True)
    (OUT / "report.json").write_text(json.dumps(reps, indent=1) + "\n")


if __name__ == "__main__":
    main()
