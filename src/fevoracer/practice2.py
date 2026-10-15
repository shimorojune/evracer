"""Practice that chases lap time once the driver gets round: survival first, then pace (own driving only).

Phase 1 is practice.py's survival practice (small antithetic populations, a lap from an official start plus
stints from random points, progress per second with a crash penalty) until the best driver laps all three
official starts.

Phase 2, pace, from then on:

* bigger populations (`pace_pairs` antithetic pairs); every candidate drives the same official start (a different
  one each step) and its fitness is that lap's time; a crash ranks below every finished lap (ordered by how far it
  got), the specialists' rule that a crash disqualifies;
* the mean driver then drives all three official starts; a candidate faster than the best so far must also lap all
  three before it is kept;
* every `imitate_every` steps a short self-imitation pass: the step's fastest clean lap is driven again with its
  own actions recorded, and the mean driver is fitted towards them (no expert labels). The result is kept only if
  its official result ranks at least as well as the mean's.

No smoothing is applied (the 0.05 s max-loss rule only matters for smoothing). Every metre driven counts towards
the budget in laps; the best driver is written at each checkpoint to winners/practice2/<circuit>-<laps>.json.

    uv run python -m fevoracer.practice2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

import fevo

from . import general
from .dagger import Student, fit
from .general import EXPERT, MASKED
from .general_es import unflatten
from .line import racing_line
from .paths import WINNERS
from .practice import GENERALS, TRACKS, drive, fmt, official, rank
from .racer import LineExpert
from .smooth import flatten

CHECKPOINTS = (0, 20, 100, 500, 2000)
OUT = WINNERS / "practice2"


def pace_rank(laps: list[tuple[str, int, float]]) -> tuple:
    """For unlock < 3: more finished starts first, then the mean time of the starts that finish."""
    fin = [s for st, s, _ in laps if st == "finished"]
    return (len(fin), -(sum(fin) / len(fin)) if fin else -1e9, float(np.mean([c for _, _, c in laps])))


def own_lap(sim: fevo.Simulator, mlp: dict, seed: int) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Drive one official start step by step, recording what the driver saw and did."""
    sim.load_program(fevo.from_json(json.dumps(mlp)))
    obs, st, xs, ys = sim.reset(seed), "running", [], []
    while st == "running":
        a = sim.evaluate_program(obs)
        s, t = float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1))
        xs.append([0.0 if i in MASKED else v for i, v in enumerate(obs)])
        ys.append((s, t))
        st = sim.step(s, t)
        obs = sim.observation()
    return np.array(xs, np.float32), np.array(ys, np.float32), st, sim.steps


def profile(sim: fevo.Simulator, mlp: dict, seed: int) -> tuple[np.ndarray, float]:
    """The driver's own speed at each centre-line point of one official start (until it ends), and metres driven."""
    sim.load_program(fevo.from_json(json.dumps(mlp)))
    n = sim.track.points
    v = np.full(n, np.nan)
    sim.reset(seed)
    st, last = "running", 0
    while st == "running":
        st = sim.agent_step()
        last = sim.progress_index
        v[last % n] = sim.core.call("getVx")
    good = ~np.isnan(v)
    v = np.interp(np.arange(n), np.where(good)[0], v[good]) if good.any() else np.full(n, 30.0)
    return v, float(last)


def drive_dist(sim: fevo.Simulator, mlp: dict, lap_seed: int, starts: list, stint: int) -> tuple[float, float]:
    """Focused practice: a lap from an official start and stints; fitness is the mean distance covered (a crash
    ends a run, so reaching it sooner earns nothing), and the metres driven."""
    sim.load_program(fevo.from_json(json.dumps(mlp)))
    n = sim.track.points
    sim.reset(lap_seed)
    st = sim.run_agent_lap()
    dists = [float(n if st == "finished" else sim.progress_index)]
    for point, lat, hdg, speed in starts:
        sim.reset(1, point=point, lateral_m=lat, heading_rad=hdg, speed=speed)
        prev, dist, st = point, 0, "running"
        for _ in range(stint):
            st = sim.agent_step()
            idx = sim.progress_index
            d = idx - prev
            dist += d + n if d < -n // 2 else d - n if d > n // 2 else d
            prev = idx
            if st != "running":
                break
        dists.append(float(max(0, dist)))
    return float(np.mean(dists)), float(sum(dists))


def practise(args: tuple) -> dict:
    (spec, pairs, pace_pairs, sigma, lr, stints, stint, crash_s, budget, imitate_every, imitate_steps, seed, unlock,
     out_dir, focus) = args
    prof_for, prof = None, None
    out_dir = WINNERS / out_dir
    # With unlock < 3 the pace phase starts with a start still crashing: candidates then drive all three starts and
    # drivers rank by finished starts, then the mean time of those (pace_rank); otherwise practice2's rules.
    rk = rank if unlock >= 3 else (lambda laps: pace_rank(laps) if sum(st == "finished" for st, _, _ in laps) >= unlock else (-1,) + rank(laps))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    setup_d = json.load(open(WINNERS / f"{GENERALS[0]}.json"))["setup"]
    sim = fevo.Simulator(general.RULES)
    setup = [setup_d[k["id"]] for k in sim.rules.setup]
    sim.set_setup(setup)
    tr = general._track(spec)
    sim.load_track(tr)
    n = tr.points
    name = spec.replace("gen:", "g")
    ex = LineExpert(tr, sim.rules, setup, offsets=racing_line(tr, -0.45), **EXPERT)
    obs, st = sim.reset(1), "running"
    ex.reset()
    while st == "running":
        a = ex(sim, obs)
        st = sim.step(a[0], a[1])
        obs = sim.observation()
    expert = sim.steps if st == "finished" else None
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
    mean_laps = laps0
    used, step_n, t0 = 0.0, 0, time.time()
    report = {"track": spec, "start": start_name, "expert": expert, "points": [], "pace_from": None, "imitation": [0, 0]}
    marks = [c for c in CHECKPOINTS if 0 < c <= budget]

    def checkpoint(budget_laps: int) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}-{budget_laps}.json"
        if not path.exists():
            path.write_text(json.dumps(unflatten(base, best_x)) + "\n")
        report["points"].append({"budget": budget_laps, "used": round(used, 1), "laps": best_laps})

    def adam(g: np.ndarray) -> None:
        nonlocal x, m, v
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        s = lr * (m / (1 - b1 ** step_n)) / (np.sqrt(v / (1 - b2 ** step_n)) + eps)
        s[frozen] = 0.0
        x = x + s

    def grad(f: np.ndarray, e: np.ndarray, p: int) -> np.ndarray:
        r = np.empty(len(f))
        r[np.argsort(f)] = np.arange(len(f))
        u = r / (len(f) - 1) - 0.5
        return ((u[:p] - u[p:])[:, None] * e).sum(0) / (2 * p * sigma)

    checkpoint(0)
    while marks:
        step_n += 1
        lapped = rank(best_laps)[0] >= unlock
        if not lapped and focus:
            # Focused survival practice: stints from just before where the best driver crashes, at its own speed
            # there (0.6-1.05x) and spread across the track; fitness is distance covered. Wider search.
            crashing = [(c, i) for i, (st_, _, c) in enumerate(best_laps) if st_ != "finished"]
            _, lap_seed = min(crashing)
            fail = int(min(crashing)[0] * n)
            key = (lap_seed, fail)
            if prof_for != key:
                prof, metres = profile(sim, unflatten(base, best_x), lap_seed)
                used += metres / n
                prof_for = key
                report.setdefault("failures", []).append({"at_laps": round(used, 1), "point": fail})
            starts = []
            for _ in range(stints):
                p = (fail - int(rng.integers(50, 400))) % n
                starts.append((p, float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-0.03, 0.03)),
                               float(max(10.0, prof[p] * rng.uniform(0.6, 1.05)))))
            fpairs, fsig = focus
            e = rng.standard_normal((fpairs, len(x)))
            e[:, frozen] = 0.0
            fits = []
            for c in [x + fsig * k for k in e] + [x - fsig * k for k in e]:
                f, metres = drive_dist(sim, unflatten(base, c), lap_seed, starts, stint)
                fits.append(f)
                used += metres / n
            r_ = np.empty(len(fits))
            r_[np.argsort(fits)] = np.arange(len(fits))
            u = r_ / (len(fits) - 1) - 0.5
            adam(((u[:fpairs] - u[fpairs:])[:, None] * e).sum(0) / (2 * fpairs * fsig))
        elif not lapped:
            # Phase 1: survival practice, as practice.py.
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
            adam(grad(np.array(fits), e, pairs))
        else:
            if report["pace_from"] is None:
                report["pace_from"] = round(used, 1)
                x = best_x.copy()  # pace starts from the driver that gets round
                m, v = np.zeros_like(x), np.zeros_like(x)
            # Phase 2: pace. Same official start for every candidate; lap time, a crash ranks last.
            seed = int(rng.integers(3))
            if unlock < 3:
                e = rng.standard_normal((pace_pairs, len(x)))
                e[:, frozen] = 0.0
                cs = [x + sigma * k for k in e] + [x - sigma * k for k in e]
                fits, best_c = [], None
                for c in cs:
                    laps_c, metres = official(sim, unflatten(base, c))
                    used += metres / n
                    pr = pace_rank(laps_c)
                    fits.append(pr[0] * 1e7 + (pr[1] if pr[0] else 0) + pr[2])
                    if rk(laps_c) > rk(best_laps) and (best_c is None or rk(laps_c) > rk(best_c[1])):
                        best_c = (c, laps_c)
                adam(grad(np.array(fits), e, pace_pairs))
                if best_c is not None:
                    best_x, best_laps = best_c[0].copy(), best_c[1]
                if imitate_every and step_n % imitate_every == 0:
                    # Self-imitation on the best driver's fastest clean start.
                    fin = [(s_, i) for i, (st_, s_, _) in enumerate(best_laps) if st_ == "finished"]
                    if fin:
                        xs, ys, st, _ = own_lap(sim, unflatten(base, best_x), min(fin)[1])
                        used += 1.0
                        if st == "finished":
                            stu = Student.from_mlp(unflatten(base, x))
                            fit(stu, xs, ys, imitate_steps, 3e-4, 4.0, batch=256)
                            general._hold_mask(stu)
                            x_im = flatten(stu.to_mlp(setup_d))
                            x_im[frozen] = 0.0
                            laps_im, metres = official(sim, unflatten(base, x_im))
                            used += metres / n
                            report["imitation"][1] += 1
                            if rk(laps_im) >= rk(mean_laps):
                                x = x_im
                                report["imitation"][0] += 1
                mean_laps, metres = official(sim, unflatten(base, x))
                used += metres / n
                if rk(mean_laps) > rk(best_laps):
                    best_x, best_laps = x.copy(), mean_laps
                if report.get("three") is None and sum(st_ == "finished" for st_, _, _ in best_laps) == 3:
                    report["three"] = round(used, 1)
                while marks and used >= marks[0]:
                    checkpoint(marks.pop(0))
                continue
            e = rng.standard_normal((pace_pairs, len(x)))
            e[:, frozen] = 0.0
            cs = [x + sigma * k for k in e] + [x - sigma * k for k in e]
            fits, best_c = [], None
            for c in cs:
                sim.load_program(fevo.from_json(json.dumps(unflatten(base, c))))
                sim.reset(seed)
                st = sim.run_agent_lap()
                done = 1.0 if st == "finished" else sim.progress_index / n
                used += done
                fits.append(-sim.steps if st == "finished" else -1e7 + done)
                if st == "finished" and (best_c is None or sim.steps < best_c[1]):
                    best_c = (c, sim.steps)
            adam(grad(np.array(fits), e, pace_pairs))
            # The fastest candidate this step may be the new best, if it laps all three starts.
            if best_c is not None:
                laps, metres = official(sim, unflatten(base, best_c[0]))
                used += metres / n
                if rank(laps) > rank(best_laps):
                    best_x, best_laps = best_c[0].copy(), laps
            # Self-imitation: fit the mean driver towards this step's fastest clean lap.
            if imitate_every and best_c is not None and step_n % imitate_every == 0:
                xs, ys, st, _ = own_lap(sim, unflatten(base, best_c[0]), seed)
                used += 1.0
                if st == "finished":
                    stu = Student.from_mlp(unflatten(base, x))
                    fit(stu, xs, ys, imitate_steps, 3e-4, 4.0, batch=256)
                    general._hold_mask(stu)
                    x_im = flatten(stu.to_mlp(setup_d))
                    x_im[frozen] = 0.0
                    laps_im, metres = official(sim, unflatten(base, x_im))
                    used += metres / n
                    report["imitation"][1] += 1
                    if rank(laps_im) >= rank(mean_laps):
                        x = x_im
                        report["imitation"][0] += 1
        mean_laps, metres = official(sim, unflatten(base, x))
        used += metres / n
        if rk(mean_laps) > rk(best_laps):
            best_x, best_laps = x.copy(), mean_laps
        while marks and used >= marks[0]:
            checkpoint(marks.pop(0))
    report["steps"], report["minutes"] = step_n, round((time.time() - t0) / 60, 1)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(prog="practice2", description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", default=",".join(TRACKS))
    ap.add_argument("--pairs", type=int, default=4, help="antithetic pairs per survival step")
    ap.add_argument("--pace-pairs", type=int, default=16, help="antithetic pairs per pace step")
    ap.add_argument("--sigma", type=float, default=3e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stints", type=int, default=2)
    ap.add_argument("--stint", type=int, default=400)
    ap.add_argument("--crash-s", type=float, default=5.0)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--imitate-every", type=int, default=5)
    ap.add_argument("--imitate-steps", type=int, default=300)
    ap.add_argument("--unlock", type=int, default=3, help="official starts lapped before the pace phase starts")
    ap.add_argument("--out", default="practice2", help="winners/<out>/")
    ap.add_argument("--focus-pairs", type=int, default=0, help="focused survival practice: antithetic pairs (0: off)")
    ap.add_argument("--focus-sigma", type=float, default=6e-3)
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    specs = a.tracks.split(",")
    # Each circuit keeps the seed it has in the default list, so its survival practice matches earlier runs.
    idx = [TRACKS.index(s) if s in TRACKS else 100 + k for k, s in enumerate(specs)]
    with mp.get_context("spawn").Pool(a.workers or min(len(specs), max(1, (os.cpu_count() or 2) - 1))) as pool:
        reps = pool.map(practise, [(s, a.pairs, a.pace_pairs, a.sigma, a.lr, a.stints, a.stint, a.crash_s, a.budget,
                                    a.imitate_every, a.imitate_steps, a.seed + i, a.unlock, a.out,
                                    (a.focus_pairs, a.focus_sigma) if a.focus_pairs else None)
                                   for i, s in zip(idx, specs)])
    print(f"{'circuit':<12} {'start driver':<28} " + "  ".join(f"{c:>22}" for c in CHECKPOINTS if c <= a.budget) + "  expert")
    for rp in reps:
        cells = [f"{fmt(p['laps'], rp['expert']):>22}" for p in rp["points"]]
        print(f"{rp['track']:<12} {rp['start']:<28} " + "  ".join(cells)
              + f"  {fevo.format_lap_time(rp['expert']) if rp['expert'] else 'fails'}"
              + f"   (pace from {rp['pace_from']} laps; imitation kept {rp['imitation'][0]}/{rp['imitation'][1]}; "
              f"{rp['steps']} steps, {rp['minutes']} min; all three starts from {rp.get('three')} laps)", flush=True)
    out = WINNERS / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(reps, indent=1) + "\n")


if __name__ == "__main__":
    main()
