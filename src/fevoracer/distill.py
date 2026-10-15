"""Distillation: many practised drivers, one general network.

1. Practice (practice2 with practice4's focused survival, pace from 2 of 3 starts lapped) on `tracks` fresh training
   circuits: generated, slow-corner and mixed styles from the training seed range. Never a held-out circuit.
2. The practised drivers that lap their circuit teach one general network, DAgger style: on each circuit the student
   drives (the teacher drives a share `beta` of the steps, falling each round), from the official starts and from
   random points, and that circuit's teacher labels every state. No expert, no racing line.
3. Each round the student is scored on the held-out circuits as general.py scores them (five laps each) and kept
   when it finishes more held-out laps (mean completion breaks ties).

    uv run python -m fevoracer.distill --tracks 40 --budget 500 --rounds 8
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
from .general import HELDOUT_GEN, MASKED, _mask, summarise
from .paths import CHECKPOINTS, WINNERS
from . import practice2, realtracks

OUT = "distill-teachers"


class Net:
    """A tanh MLP from its agent file, evaluated in numpy (switching the simulator's loaded program every step to
    ask the teacher would reload a whole network each time)."""

    def __init__(self, mlp: dict):
        dims = mlp["layers"]
        self.w = [np.array(w).reshape(dims[i + 1], dims[i]) for i, w in enumerate(mlp["weights"])]
        self.b = [np.array(b) for b in mlp["biases"]]

    def __call__(self, x) -> np.ndarray:
        h = np.asarray(x, dtype=float)
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            h = w @ h + b
            if i < len(self.w) - 1:
                h = np.tanh(h)
        return h


def training_specs(count: int, seed: int) -> list[str]:
    """Fresh training circuits: generated, slow-corner and mixed styles from the training seed range (1-800000; the
    held-out generated circuits are 900000 and up)."""
    rng = np.random.default_rng(seed)
    kinds = ["gen", "slow", "mix"]
    return [f"{kinds[i % 3]}:{int(s)}" for i, s in enumerate(rng.integers(1, 800_000, count))]


def teach(args: tuple) -> tuple[np.ndarray, np.ndarray, int]:
    """Student-driven states on one circuit, labelled by that circuit's teacher."""
    spec, teacher_path, student_mlp, beta, stints, stint, seed = args
    sim = general._SIM
    tr = general._track(spec)
    sim.load_track(tr)
    n = tr.points
    teacher = Net(json.loads(open(teacher_path).read()))
    sim.load_program(fevo.from_json(json.dumps(student_mlp)))
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    runs = [("start", int(s)) for s in range(3)] + [("point", int(rng.integers(n))) for _ in range(stints)]
    for kind, v in runs:
        if kind == "start":
            obs, steps = sim.reset(v), 12000
        else:
            obs, steps = sim.reset(1, point=v, lateral_m=float(rng.uniform(-1.5, 1.5)), heading_rad=float(rng.uniform(-0.03, 0.03)),
                                   speed=float(rng.uniform(25, 75))), stint
        st = "running"
        for _ in range(steps):
            a_t = teacher(obs)
            xs.append(_mask(obs))
            ys.append((float(np.clip(a_t[0], -1, 1)), float(np.clip(a_t[1], -1, 1))))
            if rng.random() >= beta:
                a = sim.evaluate_program(obs)
            else:
                a = a_t
            st = sim.step(float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1)))
            obs = sim.observation()
            if st != "running":
                break
    return np.array(xs, np.float32), np.array(ys, np.float32), len(xs)


def main() -> None:
    ap = argparse.ArgumentParser(prog="distill", description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", type=int, default=40)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--init", default="winners/general-solo4-es.json")
    ap.add_argument("--name", default="general-solo4-distill")
    ap.add_argument("--stints", type=int, default=8)
    ap.add_argument("--stint", type=int, default=500)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--teachers", default=OUT, help="winners/<teachers>/ holds the practised drivers")
    ap.add_argument("--skip-practice", action="store_true", help="reuse the teachers already in winners/distill-teachers")
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    workers = a.workers or max(1, (os.cpu_count() or 2) - 1)
    specs = training_specs(a.tracks, a.seed)
    held = [f"gen:{s}" for s in HELDOUT_GEN] + realtracks.names("test") + ["spielberg"]
    assert not set(specs) & set(held)
    t0 = time.time()
    if not a.skip_practice:
        print(f"Practising on {len(specs)} training circuits, {a.budget} laps each", flush=True)
        with mp.get_context("spawn").Pool(workers) as pool:
            reps = pool.map(practice2.practise, [(s, 4, 16, 3e-3, 1e-3, 4, 600, 5.0, a.budget, 5, 300, a.seed + i, 2, a.teachers, (16, 6e-3))
                                                 for i, s in enumerate(specs)])
        (WINNERS / a.teachers / "report.json").write_text(json.dumps(reps, indent=1) + "\n")
        print(f"practice done ({(time.time() - t0) / 60:.0f} min)", flush=True)
    reps = json.loads((WINNERS / a.teachers / "report.json").read_text())
    teachers = []
    for r in reps:
        laps = r["points"][-1]["laps"]
        fin = sum(st == "finished" for st, _, _ in laps)
        if fin >= 2:
            name = r["track"].replace("gen:", "g")
            teachers.append((r["track"], str(WINNERS / a.teachers / f"{name}-{r['points'][-1]['budget']}.json")))
    print(f"{len(teachers)} of {len(reps)} practised drivers lap their circuit (2 or 3 of 3 starts) and teach", flush=True)

    init = json.loads(open(a.init).read())
    setup_d = init["setup"]
    rules = fevo.load_rules(general.RULES)
    setup = [setup_d[k["id"]] for k in rules.setup]
    torch.manual_seed(a.seed)
    student = Student.from_mlp(init)
    general._hold_mask(student)
    X, Y = [], []
    out = WINNERS / f"{a.name}.json"
    with mp.get_context("spawn").Pool(workers, initializer=general._init, initargs=(setup,)) as pool:
        ev0 = pool.map(general._official, [(h, init, False) for h in held])
        line, score, fin, total = summarise(ev0)
        best = (fin, score)
        print(f"start {a.init}: held out {line} | completed {score:.1%}, laps finished {fin}/{total}", flush=True)
        for r in range(a.rounds):
            beta = max(0.0, 0.5 - 0.1 * r)
            mlp = student.to_mlp(setup_d)
            res = pool.map(teach, [(s, f, mlp, beta, a.stints, a.stint, a.seed * 1000 + r * 100 + i) for i, (s, f) in enumerate(teachers)])
            for xs, ys, _ in res:
                X.append(xs)
                Y.append(ys)
            Xa, Ya = np.concatenate(X), np.concatenate(Y)
            loss = fit(student, Xa, Ya, a.steps, 1e-3, 4.0)
            general._hold_mask(student)
            mlp = student.to_mlp(setup_d)
            ev = pool.map(general._official, [(h, mlp, False) for h in held])
            line, score, fin, total = summarise(ev)
            mark = ""
            if (fin, score) > best:
                best = (fin, score)
                out.write_text(json.dumps(mlp) + "\n")
                mark = "  << new best"
            print(f"round {r}  beta {beta:.1f}  {len(Xa):,} states  loss {loss:.4f}  ({(time.time() - t0) / 60:.0f} min)\n"
                  f"   held out: {line}  | completed {score:.1%}, laps finished {fin}/{total}{mark}", flush=True)
            torch.save(student.state_dict(), CHECKPOINTS / f"{a.name}-student.pt")
    print(f"\nBest held-out: {best[0]} laps, completion {best[1]:.1%}: {out}", flush=True)


if __name__ == "__main__":
    main()
