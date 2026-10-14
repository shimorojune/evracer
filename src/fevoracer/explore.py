"""Exploration-heavy search for lap time: an evolution strategy over the whole network.

The polishing runs (smooth-inputs) keep one driver and accept a tiny random
nudge only if every lap still finishes and nothing gets slower; that finds
hundredths, never a different way round. This searches wider:

* a population of perturbations of the current weights each generation, in
  antithetic pairs (+e and -e), with rank-shaped fitness and an Adam step along
  the estimated gradient (OpenAI-ES): every candidate teaches something, the
  ones that crash included;
* a dense, crash-tolerant fitness: each episode scores metres of progress per
  second, a crash adds `crash_s` seconds instead of zeroing the candidate, so
  a bold change that crashes late still beats one that crawls;
* varied starts: the three official starts (whole laps, run inside the wasm)
  plus `practice` short stints from random points with random offsets,
  headings and speeds;
* acceptance as before: a network is written only when all three official
  starts finish and the median beats the best so far.

A baseline with fewer inputs than the rules give (15 for solo-4's 28) is
widened with zero weights on the new inputs, so the search starts from exactly
the baseline's laps.

    uv run python -m fevoracer.explore --rules solo-4 --baseline winners/solo3-spielberg.json --name explore
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from typing import Any

import numpy as np

import fevo

from .paths import track_dir, winner_path
from .smooth import flatten, unflatten

_SIM: fevo.Simulator | None = None
_CFG: dict[str, Any] = {}


def widen(mlp: dict[str, Any], n_in: int) -> dict[str, Any]:
    """The same network taking n_in inputs: zero weights for the extra ones (first-layer rows are [out][in])."""
    have = mlp["layers"][0]
    if have == n_in:
        return mlp
    assert have < n_in, f"baseline takes {have} inputs, rules give {n_in}"
    n_out, w = mlp["layers"][1], mlp["weights"][0]
    rows = [w[j * have : (j + 1) * have] + [0.0] * (n_in - have) for j in range(n_out)]
    return dict(mlp, layers=[n_in] + mlp["layers"][1:], weights=[sum(rows, [])] + mlp["weights"][1:])


def _init(track: str, rules: str, cfg: dict[str, Any]) -> None:
    global _SIM, _CFG
    _SIM, _CFG = fevo.Simulator(rules), cfg
    _SIM.load_track(track)


def _progress(prev: int, now: int, n: int) -> int:
    d = now - prev
    return d + n if d < -n // 2 else d - n if d > n // 2 else d


def episodes(sim: fevo.Simulator, mlp: dict[str, Any], practice: list[tuple[int, float, float, float]], cfg: dict[str, Any]) -> dict:
    """Official laps and practice stints for one network: progress rate per episode and the official result."""
    sim.load_program(fevo.from_json(json.dumps(mlp)))
    hz, n = sim.rules.agent_hz, sim.track.points
    rates, laps = [], []
    for s in range(len(sim.rules.seeds)):
        sim.reset(s)
        status = sim.run_agent_lap()
        steps = sim.steps
        dist = n if status == "finished" else max(0, sim.progress_index if sim.progress_index < n - 100 else 0)
        t = steps / hz + (0 if status == "finished" else cfg["crash_s"])
        rates.append(dist / t)
        laps.append((status, steps))
    for point, lat, hdg, speed in practice:
        sim.reset(1, point=point, lateral_m=lat, heading_rad=hdg, speed=speed)
        prev, dist, status = point, 0, "running"
        for _ in range(cfg["stint"]):
            status = sim.agent_step()
            idx = sim.progress_index
            dist += _progress(prev, idx, n)
            prev = idx
            if status != "running":
                break
        t = sim.steps / hz + (cfg["crash_s"] if status not in ("running", "finished") else 0)
        rates.append(max(0, dist) / max(t, 1e-3))
    finished = sorted(st for s, st in laps if s == "finished")
    return {"fitness": float(np.mean(rates)), "laps": laps, "clean": len(finished) == len(laps),
            "median": finished[len(finished) // 2] if finished else None}


def _eval(args: tuple[dict[str, Any], np.ndarray, list]) -> dict:
    base, x, practice = args
    return episodes(_SIM, unflatten(base, x), practice, _CFG)


def explore(track: str, rules: str, baseline: str, name: str, generations: int, pairs: int, sigma: float, lr: float,
            practice: int, stint: int, crash_s: float, workers: int | None, seed: int, fresh: bool) -> None:
    sim = fevo.Simulator(rules)
    sim.load_track(track)
    base = widen(json.load(open(baseline)), len(sim.rules.observation))
    out = winner_path(name, track)
    if os.path.abspath(out) == os.path.abspath(baseline):
        raise SystemExit(f"{out} is the baseline; pick another --name")
    cfg = {"crash_s": crash_s, "stint": stint}
    state_file = track_dir(track) / f"{name}-es.pkl"
    rng = np.random.default_rng(seed)
    x = flatten(base)
    m, v, gen0 = np.zeros_like(x), np.zeros_like(x), 0
    start = episodes(sim, base, [], cfg)
    best = start["median"] if start["clean"] else None
    if state_file.is_file() and not fresh:
        st = pickle.loads(state_file.read_bytes())
        x, m, v, gen0, best, rng = st["x"], st["m"], st["v"], st["gen"], st["best"], st["rng"]
        print(f"resumed at generation {gen0}, best {fevo.format_lap_time(best) if best else 'none'}", flush=True)
    fmt = lambda laps: " / ".join(fevo.format_lap_time(s) if st == "finished" else st for st, s in laps)  # noqa: E731
    print(f"Explore at {track} ({rules}), {len(x)} numbers, {2 * pairs} candidates a generation, sigma {sigma}, lr {lr}, "
          f"{practice} practice stints of {stint} steps, crash +{crash_s} s. Baseline {os.path.basename(baseline)}: {fmt(start['laps'])}", flush=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    b1, b2, eps = 0.9, 0.999, 1e-8
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(track, rules, cfg)) as pool:
        for gen in range(gen0 + 1, gen0 + generations + 1):
            # The same practice starts for every candidate this generation (a fair comparison), new ones next time.
            starts = []
            for _ in range(practice):
                p = int(rng.integers(sim.track.points))
                starts.append((p, float(rng.uniform(-2.5, 2.5)), float(rng.uniform(-0.05, 0.05)), float(rng.uniform(25, 75))))
            noise = rng.standard_normal((pairs, len(x)))
            cands = [x + sigma * e for e in noise] + [x - sigma * e for e in noise]
            res = pool.map(_eval, [(base, c, starts) for c in cands])
            f = np.array([r["fitness"] for r in res])
            # Rank shaping: centred ranks in [-0.5, 0.5], robust to a few huge or terrible scores.
            ranks = np.empty(len(f))
            ranks[np.argsort(f)] = np.arange(len(f))
            u = ranks / (len(f) - 1) - 0.5
            g = ((u[:pairs] - u[pairs:])[:, None] * noise).sum(0) / (2 * pairs * sigma)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            x = x + lr * (m / (1 - b1**gen)) / (np.sqrt(v / (1 - b2**gen)) + eps)
            # Acceptance: the mean network and the best candidate, on the official starts only.
            mean = episodes(sim, unflatten(base, x), [], cfg)
            mark = ""
            for label, r, params in (("mean", mean, x), ("candidate", res[int(np.argmax(f))], cands[int(np.argmax(f))])):
                if r["clean"] and (best is None or r["median"] < best):
                    best = r["median"]
                    out.write_text(json.dumps(unflatten(base, params)) + "\n")
                    mark = f"  << new best ({label})"
            clean = sum(r["clean"] for r in res)
            print(f"gen {gen:4d}  {(time.time() - t0) / 60:6.1f} min  fitness mean {f.mean():6.2f} max {f.max():6.2f} m/s  "
                  f"clean {clean:2d}/{len(res)}  mean net {fmt(mean['laps'])}  |  best {fevo.format_lap_time(best) if best else 'none'}{mark}", flush=True)
            if gen % 5 == 0 or mark:
                state_file.write_bytes(pickle.dumps({"x": x, "m": m, "v": v, "gen": gen, "best": best, "rng": rng}))
    print(f"\nBest {fevo.format_lap_time(best) if best else 'none'}: {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="explore", description="Population evolution strategy for lap time, crash-tolerant, varied starts")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("--rules", default="solo-4")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--name", default="explore", help="writes winners/<name>-<track>.json")
    ap.add_argument("-g", "--generations", type=int, default=500)
    ap.add_argument("--pairs", type=int, default=24, help="antithetic pairs per generation (population = 2 x pairs)")
    # Calibrated on the 1:11.25 solo-3 driver: at sigma 3e-4, 6 in 8 perturbed drivers still finish all
    # three starts; at 1e-3 only 1 in 8. The Adam step moves every weight by about lr a generation.
    ap.add_argument("--sigma", type=float, default=3e-4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--practice", type=int, default=4, help="random practice stints per candidate")
    ap.add_argument("--stint", type=int, default=600, help="steps per practice stint")
    ap.add_argument("--crash-s", type=float, default=5.0, help="seconds added to an episode that crashes")
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    explore(a.track, a.rules, a.baseline, a.name, a.generations, a.pairs, a.sigma, a.lr, a.practice, a.stint, a.crash_s,
            a.workers, a.seed, a.fresh)


if __name__ == "__main__":
    main()
