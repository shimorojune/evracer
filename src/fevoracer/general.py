"""A general driver: one network for any circuit, taught by the racing-line expert on many tracks.

DAgger across tracks. Each round takes a fresh batch of generated circuits
(fevoracer.trackgen) and a sample of the real training circuits
(fevoracer.realtracks), and on each one the full-information expert
(fevoracer.racer.LineExpert on that track's one-tyre racing line) labels every
state while a growing share of the driving is done by the student. The student
is a fresh MLP on solo-4's 33 inputs with the two progress inputs masked (zeroed
in the data and their first-layer weights held at zero): it has to read the
road from the rays, offsets, curvature ahead and its own motion, not remember
where it is on a lap.

Generalisation is measured every round on tracks it never trains on: fixed
held-out generated circuits and the held-out real ones (Monza, Austin, Montreal,
Spa), with Spielberg as the reference point. The best student by held-out
progress goes to winners/<name>.json.

    uv run python -m fevoracer.general --name general
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from typing import Any

import numpy as np
import torch

import fevo

from . import realtracks, trackgen
from .dagger import Student, fit
from .line import racing_line
from .paths import CHECKPOINTS, WINNERS
from .racer import LineExpert

RULES = "solo-4"
#: Inputs a general driver must not use: lap progress by gate (index 12) and by metre (the last, index 32).
MASKED = (12, 32)
EXPERT = {"line_scale": 0.9, "look_s": 0.45, "yaw_damp": 0.08, "k_cross": 1.0, "margin": 0.96, "slow_margin": 0.9}
CAUTIOUS = {**EXPERT, "line_scale": 0.8, "slow_margin": 0.85}
def _distinct_gen(first: int, count: int) -> list[int]:
    """Seeds of `count` distinct valid generated circuits from `first` on (a failed seed would repeat the next one)."""
    out, s = [], first
    while len(out) < count:
        if trackgen.generate(s) is not None:
            out.append(s)
        s += 1
    return out


HELDOUT_GEN = _distinct_gen(900_000, 8)  # generator seeds never used for training
EXCLUDE_REAL = {"catalunya", "sakhir"}  # the expert crashes at the same metre whatever its setting: data to inspect

_SIM: fevo.Simulator | None = None
_SETUP: list[float] | None = None
_CACHE: dict[str, tuple[fevo.Track, np.ndarray, dict]] = {}


def _init(setup: list[float]) -> None:
    global _SIM, _SETUP
    torch.set_num_threads(1)
    _SIM, _SETUP = fevo.Simulator(RULES), setup
    _SIM.set_setup(setup)


def _track(spec: str) -> fevo.Track:
    """gen:<seed> is the first valid generated circuit at or after that seed (most seeds fail a check)."""
    if not spec.startswith("gen:"):
        return realtracks.load(spec)
    s = int(spec[4:])
    while (t := trackgen.generate(s)) is None:
        s += 1
    return trackgen.track(t)


def _expert_for(spec: str) -> tuple[fevo.Track, LineExpert, dict]:
    """The track, its expert and the settings that lap it (default, else cautious), cached per worker."""
    if spec not in _CACHE:
        tr = _track(spec)
        off = racing_line(tr, -0.45)
        kw = EXPERT
        _SIM.load_track(tr)
        for cand in (EXPERT, CAUTIOUS):
            ex = LineExpert(tr, _SIM.rules, _SETUP, offsets=off, **cand)
            obs, st = _SIM.reset(1), "running"
            ex.reset()
            while st == "running":
                s, t = ex(_SIM, obs)
                st = _SIM.step(s, t)
                obs = _SIM.observation()
            kw = cand
            if st == "finished":
                break
        _CACHE[spec] = (tr, off, kw)
    tr, off, kw = _CACHE[spec]
    return tr, LineExpert(tr, _SIM.rules, _SETUP, offsets=off, **kw), kw


def _mask(obs: list[float]) -> list[float]:
    o = list(obs)
    for i in MASKED:
        o[i] = 0.0
    return o


def _rollouts(args: tuple) -> tuple[np.ndarray, np.ndarray, dict]:
    """One track: a lap from a random official start plus practice stints, mixing expert and student driving."""
    spec, student_mlp, beta, practice, stint, seed = args
    rng = np.random.default_rng(seed)
    tr, ex, kw = _expert_for(spec)
    sim = _SIM
    sim.load_track(tr)
    if student_mlp is not None:
        sim.load_program(fevo.from_json(json.dumps(student_mlp)))
    xs, ys, end = [], [], {}
    n = tr.points
    starts = [{"seed": int(rng.integers(3)), "steps": 12000}]
    for _ in range(practice):
        p = int(rng.integers(n))
        starts.append({"point": p, "lat": float(rng.uniform(-2.0, 2.0)), "hdg": float(rng.uniform(-0.05, 0.05)),
                       "speed": float(rng.uniform(0.6, 1.0) * ex.v[p]), "steps": stint})
    for st_ in starts:
        if "seed" in st_:
            obs = sim.reset(st_["seed"])
        else:
            obs = sim.reset(1, point=st_["point"], lateral_m=st_["lat"], heading_rad=st_["hdg"], speed=st_["speed"])
        ex.reset()
        status = "running"
        for _ in range(st_["steps"]):
            a_exp = ex(sim, obs)
            xs.append(_mask(obs))
            ys.append(a_exp)
            if student_mlp is not None and rng.random() >= beta:
                a = sim.evaluate_program(obs)
            else:
                a = a_exp
            status = sim.step(float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1)))
            obs = sim.observation()
            if status != "running":
                break
        if "seed" in st_:
            end = {"status": status, "progress": sim.progress_index / n, "expert": "default" if kw is EXPERT else "cautious"}
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), end


def _official(args: tuple) -> dict:
    """The student's official laps on one track (all three starts), and the expert's for comparison."""
    spec, student_mlp, with_expert = args
    tr, ex, _ = _expert_for(spec)
    sim = _SIM
    sim.load_track(tr)
    out = {"track": spec, "laps": [], "expert": []}
    if student_mlp is not None:
        sim.load_program(fevo.from_json(json.dumps(student_mlp)))
        for s in range(3):
            sim.reset(s)
            st = sim.run_agent_lap()
            out["laps"].append((st, sim.steps, sim.progress_index / tr.points))
    for s in range(3 if with_expert else 0):
        obs, st = sim.reset(s), "running"
        ex.reset()
        while st == "running":
            a = ex(sim, obs)
            st = sim.step(a[0], a[1])
            obs = sim.observation()
        out["expert"].append((st, sim.steps))
    return out


def masked_student(hidden: list[int]) -> Student:
    st = Student(hidden, len(fevo.load_rules(RULES).observation))
    with torch.no_grad():
        first = [m for m in st.net if isinstance(m, torch.nn.Linear)][0]
        first.weight[:, list(MASKED)] = 0.0
    return st


def _hold_mask(st: Student) -> None:
    with torch.no_grad():
        [m for m in st.net if isinstance(m, torch.nn.Linear)][0].weight[:, list(MASKED)] = 0.0


def summarise(res: list[dict], hz: int = 60) -> tuple[str, float]:
    """One line per track group and a single score: mean share of the lap completed over all held-out laps."""
    parts, prog = [], []
    for r in res:
        done = [s for st, s, _ in r["laps"] if st == "finished"]
        ex = sorted(s for st, s in r["expert"] if st == "finished")
        prog += [1.0 if st == "finished" else p for st, _, p in r["laps"]]
        lab = r["track"].replace("gen:", "g")
        if len(done) == 3:
            med = sorted(done)[1]
            parts.append(f"{lab} {fevo.format_lap_time(med)}" + (f" ({100 * (med / ex[1] - 1):+.0f}%)" if len(ex) == 3 else ""))
        else:
            parts.append(f"{lab} {len(done)}/3 ({np.mean([1.0 if st == 'finished' else p for st, _, p in r['laps']]):.0%})")
    return "  ".join(parts), float(np.mean(prog)) if prog else 0.0


def train(name: str, rounds: int, gen_per_round: int, real_per_round: int, practice: int, stint: int, hidden: list[int],
          steps: int, lr: float, workers: int | None, seed: int, setup_from: str) -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rules = fevo.load_rules(RULES)
    setup_d = json.load(open(setup_from))["setup"]
    setup = [setup_d[k["id"]] for k in rules.setup]
    student = masked_student(hidden)
    real_train = [n for n in realtracks.names("train") if n not in EXCLUDE_REAL]
    held = [f"gen:{s}" for s in HELDOUT_GEN] + realtracks.names("test") + ["spielberg"]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    best, out = -1.0, WINNERS / f"{name}.json"
    t0 = time.time()
    print(f"General driver on {RULES}: student {[len(rules.observation), *hidden, 2]} with inputs {MASKED} masked; per round {gen_per_round} new "
          f"generated + {real_per_round} of {len(real_train)} real training circuits; held out {held}", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(setup,)) as pool:
        base = pool.map(_official, [(h, None, True) for h in held])
        expert_laps = {r["track"]: r["expert"] for r in base}
        print("expert on held-out tracks: " + "  ".join(
            f"{r['track'].replace('gen:', 'g')} {fevo.format_lap_time(sorted(s for st, s in r['expert'] if st == 'finished')[1]) if sum(st == 'finished' for st, _ in r['expert']) >= 2 else 'fails'}"
            for r in base), flush=True)
        for r in range(rounds + 1):
            beta = 1.0 if r == 0 else max(0.0, 0.6 - 0.1 * r)
            specs = [f"gen:{int(s)}" for s in rng.integers(1, 800_000, gen_per_round)] + list(rng.choice(real_train, real_per_round, replace=False))
            mlp = None if r == 0 else student.to_mlp(setup_d)
            res = pool.map(_rollouts, [(s, mlp, beta, practice, stint, int(rng.integers(1 << 30))) for s in specs])
            for xs, ys, _ in res:
                X.append(xs)
                Y.append(ys)
            # Keep the most recent ~3M states so the data follows the student without growing without bound.
            while sum(len(x) for x in X) > 3_000_000:
                X.pop(0)
                Y.pop(0)
            loss = fit(student, np.concatenate(X), np.concatenate(Y), steps, lr, 4.0)
            _hold_mask(student)
            mlp = student.to_mlp(setup_d)
            ev = pool.map(_official, [(h, mlp, False) for h in held])
            for e in ev:
                e["expert"] = expert_laps[e["track"]]
            line, score = summarise(ev)
            mark = ""
            if score > best:
                best = score
                out.write_text(json.dumps(mlp) + "\n")
                mark = "  << new best"
            ends = [e for _, _, e in res]
            train_done = sum(e.get("status") == "finished" for e in ends)
            print(f"round {r:2d}  beta {beta:.1f}  {sum(len(x) for x in X):9,d} states  loss {loss:.4f}  training laps finished "
                  f"{train_done}/{len(ends)}  ({(time.time() - t0) / 60:.1f} min)\n   held out: {line}  | completed {score:.1%}{mark}",
                  flush=True)
            torch.save(student.state_dict(), CHECKPOINTS / f"{name}-student.pt")
    print(f"\nBest held-out completion {best:.1%}: {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="general", description="Train one driver for any circuit (DAgger across tracks)")
    ap.add_argument("--name", default="general", help="writes winners/<name>.json")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--gen", type=int, default=24, help="new generated circuits per round")
    ap.add_argument("--real", type=int, default=6, help="real training circuits per round")
    ap.add_argument("--practice", type=int, default=12, help="practice stints per circuit and round")
    ap.add_argument("--stint", type=int, default=500)
    ap.add_argument("--hidden", default="192,128")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--setup-from", default="winners/solo3-spielberg.json", help="the car setup the driver uses everywhere")
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    train(a.name, a.rounds, a.gen, a.real, a.practice, a.stint, [int(h) for h in a.hidden.split(",")], a.steps, a.lr,
          a.workers, a.seed, a.setup_from)


if __name__ == "__main__":
    main()
