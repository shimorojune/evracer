"""Smooth inputs: fine-tune the practice winner on lap time plus slide time.

PPO with a slide/jerk cost broke the 1:20 driver on its first updates (and so
did a control run with no cost at all: any PPO update does). This skips the
reward and the gradient and scores what we want directly, on the three
official laps:

  median lap (s) + slide × (lock-up + spin × wheelspin seconds per lap)
                 + slower × (seconds slower than the baseline lap)

                 + pedal × (throttle travel per lap)

and a lap more than max_loss slower than the baseline doesn't count at all.
Throttle travel is how far the throttle side of the pedal moves in a lap, up
and down: F1 drivers feed the power in once per corner exit, the 1:19.83 car
saws at it (78 full-pedal widths a lap, about 600 direction changes).
Wheelspin gets its own weight because the first run (from the practice
winner, equal weights) cut lock-up by a third but let wheelspin creep up:
getting out of corners faster with a bit more spin scored as a win.

The driver sits on a knife edge: moving every weight by 0.0002 at random
already crashes four candidates in five, so recombining search (CMA-ES)
averages crashers into the mean. Instead an elitist (1+λ) evolution strategy
keeps the best driver so far, tries λ random nudges of it each generation and
keeps one only if it scores better; the step grows after a success and
shrinks after a miss. After restart_after generations without a better
driver the step goes back to its starting size (the first run stalled for
200 generations at the smallest step, then crept on).

Lock-up is any tyre sliding while braking; wheelspin is the rears sliding on
throttle (the same split as the live viewer). The setup stays as the practice
session found it. The baseline file is never written; the best candidate goes
to winners/<name>-<track>.json, and the search state to
checkpoints/<track>/<name>-es.pkl so a stopped run resumes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import dataclass
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
    laps, lock, spin, pedal = [], [], [], []
    for seed in range(len(sim.rules.seeds)):
        lap = sim.run_lap(seed, trace=True)
        t = lap.trace
        f, r, th = t["frontSlide"], t["rearSlide"], t["throttle"]
        tp = np.clip(th, 0.0, 1.0)
        pedal.append(float(np.abs(np.diff(tp)).sum()))
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
        "pedal": float(np.mean([pedal[i] for i in done])),
    }


@dataclass
class Score:
    slide: float = 0.3  # per second of lock-up per lap
    spin: float = 1.0  # wheelspin weight relative to lock-up
    slower: float = 5.0  # per second slower than the baseline lap
    max_loss: float | None = None  # laps slower than baseline + this don't count
    pedal: float = 0.0  # per unit of throttle travel per lap

    def __call__(self, m: dict[str, Any], base_lap: float) -> float:
        if not m["ranked"]:
            return UNRANKED + 30.0 * (1 - m["gates"] / max(1, m["gate_count"]))
        if self.max_loss is not None and m["lap_s"] > base_lap + self.max_loss + 1e-9:
            return UNRANKED / 2 + m["lap_s"]
        return (m["lap_s"] + self.slide * (m["lock_s"] + self.spin * m["spin_s"])
                + self.slower * max(0.0, m["lap_s"] - base_lap) + self.pedal * m.get("pedal", 0.0))

    def describe(self, base: str) -> str:
        cap = f"; laps slower than {base} + {self.max_loss} s don't count" if self.max_loss is not None else ""
        return (f"score = lap + {self.slide} x (lock-up + {self.spin} x wheelspin seconds)"
                f" + {self.slower} x seconds slower than {base} + {self.pedal} x throttle travel{cap}")


def _eval(args: tuple[dict[str, Any], np.ndarray]) -> dict[str, Any]:
    base, x = args
    return measure(unflatten(base, x), _SIM)


def fmt(m: dict[str, Any]) -> str:
    lap = m["lap_time"] or f"unranked (gate {m['gates']})"
    return f"{lap}  lock-up {m['lock_s']:.2f} s  wheelspin {m['spin_s']:.2f} s  throttle travel {m.get('pedal', 0):.1f}"


def throttle_mask(mlp: dict[str, Any]) -> np.ndarray:
    """1 for the numbers that only drive the throttle: the last layer's throttle row and bias.

    Everything before the last layer feeds steering too, so it stays frozen:
    the network steers exactly as before for any given input."""
    sizes = [len(w) for w in mlp["weights"]] + [len(b) for b in mlp["biases"]]
    off = np.cumsum([0] + sizes)
    nl, h = len(mlp["weights"]), mlp["layers"][-2]
    mask = np.zeros(off[-1])
    mask[off[nl - 1] + h : off[nl - 1] + 2 * h] = 1  # weights are row-major (outputs x inputs); row 1 is throttle
    mask[off[2 * nl - 1] + 1] = 1
    return mask


def smooth(track: str, baseline: str | None, name: str, generations: int, sigma: float, popsize: int, sc: Score,
           restart_after: int, workers: int | None, seed: int, fresh: bool, only: str = "all") -> None:
    src = baseline or str(winner_path("practice", track))
    base = json.loads(open(src).read())
    out = winner_path(name, track)
    if os.path.abspath(out) == os.path.abspath(src):
        raise SystemExit(f"{out} is the baseline; pick another --name")
    state_file = track_dir(track) / f"{name}-es.pkl"
    sigma0 = sigma
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    base_m = measure(base, track=track)
    base_lap = base_m["lap_s"]
    print(f"Smooth inputs at {track}. Baseline {os.path.basename(src)}: {fmt(base_m)}", flush=True)
    print(sc.describe(base_m["lap_time"]), flush=True)

    x0 = flatten(base)
    mask = throttle_mask(base) if only == "throttle" else np.ones(len(x0))
    print(f"tuning {int(mask.sum())} of {len(x0)} numbers ({only})", flush=True)
    rng = np.random.default_rng(seed)
    if state_file.is_file() and not fresh:
        st = pickle.loads(state_file.read_bytes())
        best_x, best_m, sigma, gen0, rng = st["best_x"], st["best_m"], st["sigma"], st["gen"], st["rng"]
        print(f"resumed at generation {gen0}, sigma {sigma:.5f}, best {fmt(best_m)}", flush=True)
    else:
        best_x, best_m, gen0 = x0, base_m, 0
    best_s = sc(best_m, base_lap)
    misses = 0

    def save(gen: int) -> None:
        state_file.write_bytes(pickle.dumps({"best_x": best_x, "best_m": best_m, "sigma": sigma, "gen": gen, "rng": rng}))

    t0 = time.time()
    gen = gen0
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(track,)) as pool:
        for gen in range(gen0 + 1, gen0 + generations + 1):
            xs = [best_x + sigma * mask * rng.standard_normal(len(best_x)) for _ in range(popsize)]
            ms = pool.map(_eval, [(base, x) for x in xs])
            ss = [sc(m, base_lap) for m in ms]
            i = int(np.argmin(ss))
            mark = ""
            if ss[i] < best_s:
                best_s, best_x, best_m = ss[i], xs[i], ms[i]
                out.write_text(json.dumps(unflatten(base, best_x)) + "\n")
                sigma = min(SIGMA_MAX, sigma * 1.2)
                mark, misses = "  << new best", 0
            else:
                sigma = max(SIGMA_MIN, sigma * 0.95)
                misses += 1
                if misses >= restart_after:
                    sigma, misses, mark = sigma0, 0, "  << restart step"

            ranked = sum(m["ranked"] for m in ms)
            print(f"gen {gen:4d}  {(time.time() - t0) / 60:6.1f} min  sigma {sigma:.5f}  ranked {ranked:2d}/{len(ms)}"
                  f"  gen best {fmt(ms[i])}  |  best {fmt(best_m)}  score {best_s:.3f}{mark}", flush=True)
            if gen % 10 == 0 or "best" in mark:
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
