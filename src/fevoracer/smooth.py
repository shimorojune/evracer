"""Smooth inputs: fine-tune the practice winner on lap time plus slide time.

PPO with a slide/jerk cost broke the 1:20 driver on its first updates (and so
did a control run with no cost at all: any PPO update does). This skips the
reward and the gradient and scores what we want directly, on the three
official laps:

  median lap (s) + slide × (lock-up + wheelspin seconds per lap)
                 + slower × (seconds slower than the baseline lap)

The driver sits on a knife edge: moving every weight by 0.0002 at random
already crashes four candidates in five, so recombining search (CMA-ES)
averages crashers into the mean. Instead an elitist (1+λ) evolution strategy
keeps the best driver so far, tries λ random nudges of it each generation and
keeps one only if it scores better; the step grows after a success and
shrinks after a miss.

Lock-up is any tyre sliding while braking; wheelspin is the rears sliding on
throttle (the same split as the live viewer). The setup stays as the practice
session found it. The baseline file is never written; the best candidate goes
to winners/smooth-<track>.json, and the search state to checkpoints/<track>/
so a stopped run resumes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import time
from typing import Any

import numpy as np

import fevo

from .paths import WINNERS, track_dir, winner_path


UNRANKED = 1000.0
#: Step size bounds: the driver is so close to the edge that 0.0002 already crashes most candidates.
SIGMA_MIN, SIGMA_MAX = 5e-5, 3e-3

_SIM: fevo.Simulator | None = None
_TRACK = ""


def _init(track: str) -> None:
    global _SIM, _TRACK
    _SIM, _TRACK = fevo.Simulator(), track
    _SIM.configure()
    _SIM.load_track(track)


def flatten(mlp: dict[str, Any]) -> np.ndarray:
    return np.concatenate([np.array(w) for w in mlp["weights"]] + [np.array(b) for b in mlp["biases"]])


def unflatten(base: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    out = dict(base)
    i, ws, bs = 0, [], []
    for w in base["weights"]:
        ws.append(x[i : i + len(w)].tolist())
        i += len(w)
    for b in base["biases"]:
        bs.append(x[i : i + len(b)].tolist())
        i += len(b)
    out["weights"], out["biases"] = ws, bs
    return out


def measure(mlp: dict[str, Any], sim: fevo.Simulator | None = None, track: str | None = None) -> dict[str, Any]:
    """Official laps with lock-up and wheelspin seconds (mean per finished lap)."""
    if sim is None:
        sim = fevo.Simulator()
        sim.configure()
        sim.load_track(track or _TRACK)
    sim.load_program(fevo.from_json(json.dumps(mlp)))
    laps, lock, spin = [], [], []
    for seed in range(len(sim.rules.seeds)):
        lap = sim.run_lap(seed, trace=True)
        t = lap.trace
        f, r, th = t["frontSlide"], t["rearSlide"], t["throttle"]
        lock.append(sum(1 for a, b, c in zip(f, r, th) if c < 0 and (a > 0 or b > 0)) / 60)
        spin.append(sum(1 for b, c in zip(r, th) if c >= 0 and b > 0) / 60)
        laps.append(lap)
    res = fevo.summarize_official(laps, sim.rules.agent_hz)
    done = [i for i, l in enumerate(laps) if l.finished] or [0, 1, 2]
    return {
        "ranked": res.ranked,
        "lap_s": res.median_steps / 60 if res.ranked else None,
        "lap_time": res.lap_time,
        "gates": sorted(l.gates_passed for l in laps)[1],
        "gate_count": laps[0].gate_count,
        "lock_s": float(np.mean([lock[i] for i in done])),
        "spin_s": float(np.mean([spin[i] for i in done])),
    }


def score(m: dict[str, Any], base_lap: float, slide: float, slower: float) -> float:
    if not m["ranked"]:
        return UNRANKED + 30.0 * (1 - m["gates"] / max(1, m["gate_count"]))
    return m["lap_s"] + slide * (m["lock_s"] + m["spin_s"]) + slower * max(0.0, m["lap_s"] - base_lap)


def _eval(args: tuple[dict[str, Any], np.ndarray]) -> dict[str, Any]:
    base, x = args
    return measure(unflatten(base, x), _SIM)


def fmt(m: dict[str, Any]) -> str:
    lap = m["lap_time"] or f"unranked (gate {m['gates']})"
    return f"{lap}  lock-up {m['lock_s']:.2f} s  wheelspin {m['spin_s']:.2f} s"


def smooth(track: str, baseline: str | None, generations: int, sigma: float, popsize: int | None, slide: float,
           slower: float, workers: int | None, seed: int, fresh: bool) -> None:
    src = baseline or str(winner_path("practice", track))
    base = json.loads(open(src).read())
    out = winner_path("smooth", track)
    state_file = track_dir(track) / "smooth-cma.pkl"
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    base_m = measure(base, track=track)
    base_lap = base_m["lap_s"]
    print(f"Smooth inputs at {track}. Baseline {os.path.basename(src)}: {fmt(base_m)}", flush=True)
    print(f"score = lap + {slide} x slide seconds + {slower} x seconds slower than {base_m['lap_time']}", flush=True)

    x0 = flatten(base)
    rng = np.random.default_rng(seed)
    if state_file.is_file() and not fresh:
        st = pickle.loads(state_file.read_bytes())
        best_x, best_m, sigma, gen0, rng = st["best_x"], st["best_m"], st["sigma"], st["gen"], st["rng"]
        print(f"resumed at generation {gen0}, sigma {sigma:.5f}, best {fmt(best_m)}", flush=True)
    else:
        best_x, best_m, gen0 = x0, base_m, 0
    best_s = score(best_m, base_lap, slide, slower)

    def save(gen: int) -> None:
        state_file.write_bytes(pickle.dumps({"best_x": best_x, "best_m": best_m, "sigma": sigma, "gen": gen, "rng": rng}))

    t0 = time.time()
    gen = gen0
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(track,)) as pool:
        for gen in range(gen0 + 1, gen0 + generations + 1):
            xs = [best_x + sigma * rng.standard_normal(len(best_x)) for _ in range(popsize)]
            ms = pool.map(_eval, [(base, x) for x in xs])
            ss = [score(m, base_lap, slide, slower) for m in ms]
            i = int(np.argmin(ss))
            mark = ""
            if ss[i] < best_s:
                best_s, best_x, best_m = ss[i], xs[i], ms[i]
                out.write_text(json.dumps(unflatten(base, best_x)) + "\n")
                sigma = min(SIGMA_MAX, sigma * 1.2)
                mark = "  << new best"
            else:
                sigma = max(SIGMA_MIN, sigma * 0.95)
            ranked = sum(m["ranked"] for m in ms)
            print(f"gen {gen:4d}  {(time.time() - t0) / 60:6.1f} min  sigma {sigma:.5f}  ranked {ranked:2d}/{len(ms)}"
                  f"  gen best {fmt(ms[i])}  |  best {fmt(best_m)}  score {best_s:.3f}{mark}", flush=True)
            if gen % 10 == 0 or mark:
                save(gen)
    save(gen)
    print(f"\nBaseline {fmt(base_m)}\nBest     {fmt(best_m)}\nUpload file: {out}", flush=True)


def compare() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="smooth-compare", description="Lap time, lock-up and wheelspin of agent files")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    a = ap.parse_args()
    files = a.files or [str(winner_path("practice", a.track)), str(winner_path("smooth", a.track))]
    for f in files:
        if os.path.isfile(f):
            print(f"{os.path.relpath(f, WINNERS.parent):<36} {fmt(measure(json.loads(open(f).read()), track=a.track))}")
