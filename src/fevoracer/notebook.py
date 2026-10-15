"""Prototype: learning a track the way a driver does, with a notebook of braking points instead of new weights.

The general driver's weights stay frozen; it steers and drives. A per-track notebook (Python side only, keyed by
position on the lap) holds one note per braking zone: how many metres later (positive) or earlier (negative) than
its own instinct the driver should brake there. The wrapper applies a note the way a driver uses a braking marker:

* later by s metres: from where the driver would start braking until s metres further on, keep the throttle open;
  after that the driver brakes as it likes (it now sees the corner closer and brakes harder);
* earlier by s metres: brake from s metres before the zone's start, then the driver takes over.

Practice, from the driver's own laps only (no expert, no racing line):

* lap 1 is the driver alone; every place it starts braking becomes a zone with a note of 0 m;
* each lap tries a change at `tries` zones: `step` metres later (largest remaining step first). A change is kept
  if the lap is clean and that zone (from 150 m before its braking point to 150 m before the next zone's) got
  quicker; otherwise it is put back and that zone's step is halved, as a driver who overdid it eases back;
* a crash in a zone that was not changed that lap moves its note `crash_m` metres earlier.

Readable notes: "1.38 km: brake 15 m later". (A first version with target speeds per 25 m stretch, forcing the
throttle towards them, made laps slower and crashed more: forcing throttle mid-corner breaks a driver that is
already at its own limit. Braking points are what a driver actually adjusts.)

    uv run python -m fevoracer.notebook
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

import fevo

from . import general
from .general import EXPERT
from .line import racing_line
from .paths import WINNERS
from .practice import GENERALS, official, rank
from .racer import LineExpert

LEAD = 150  # metres before a braking point that belong to its zone


class Notebook:
    def __init__(self, n: int, points: list[int]):
        self.n = n
        self.points = sorted(points)  # where the driver first started braking, on its own
        self.shift = np.zeros(len(self.points))  # metres: positive = brake later

    def zone_of(self, j: int) -> int:
        """The zone whose stretch contains point j (stretches start LEAD metres before each braking point)."""
        k = len(self.points)
        for i, p in enumerate(self.points):
            span = (self.points[(i + 1) % k] - p) % self.n or self.n
            if (j - (p - LEAD)) % self.n < span:
                return i
        return k - 1

    def pedal(self, j: int, own: float) -> float:
        for i, p in enumerate(self.points):
            s = self.shift[i]
            d = (j - p + self.n // 2) % self.n - self.n // 2  # metres past the braking point (negative: before)
            if s > 0 and -60 <= d < s and own < 0:
                return 1.0  # not yet: my marker is further on
            if s < 0 and s <= d < 0:
                return min(own, -0.7)  # earlier: brake from my new marker
        return own


def run_lap(sim: fevo.Simulator, nb: Notebook | None, seed: int) -> dict:
    """One lap from an official start; steps spent at each point, where braking started, how it ended."""
    n = sim.track.points
    obs, st = sim.reset(seed), "running"
    at = np.zeros(n)
    brake_at, braking = [], False
    while st == "running":
        j = sim.progress_index % n
        steer, own = sim.evaluate_program(obs)
        own = float(np.clip(own, -1, 1))
        thr = nb.pedal(j, own) if nb else own
        if thr < -0.05 and not braking:
            brake_at.append(j)
        braking = thr < -0.05
        at[j] += 1
        st = sim.step(float(np.clip(steer, -1, 1)), thr)
        obs = sim.observation()
    return {"status": st, "steps": sim.steps, "point": sim.progress_index, "at": at, "brake_at": brake_at}


def zones_from(brake_at: list[int], n: int, gap: int = 250) -> list[int]:
    """Braking points at least `gap` metres apart (a zone's later dabs belong to it)."""
    out: list[int] = []
    for p in sorted(brake_at):
        if not out or p - out[-1] > gap:
            out.append(p)
    if len(out) > 1 and (out[0] + n - out[-1]) <= gap:
        out.pop()
    return out


class Practice:
    def __init__(self, nb: Notebook, tries: int, step: float, crash_m: float, rng: np.random.Generator):
        self.nb, self.tries, self.crash_m, self.rng = nb, tries, crash_m, rng
        k = len(nb.points)
        self.step = np.full(k, step)
        self.best = np.full(k, np.inf)
        self.trial: list[tuple[int, float]] = []
        self.kept = 0

    def zone_time(self, lap: dict, z: int) -> float:
        nb = self.nb
        a = (nb.points[z] - LEAD) % nb.n
        span = (nb.points[(z + 1) % len(nb.points)] - nb.points[z]) % nb.n or nb.n
        return float(lap["at"][(a + np.arange(span)) % nb.n].sum())

    def remember(self, lap: dict) -> None:
        for z in range(len(self.nb.points)):
            self.best[z] = min(self.best[z], self.zone_time(lap, z))

    def plan(self) -> None:
        order = sorted(range(len(self.nb.points)), key=lambda z: (-self.step[z], self.rng.random()))
        self.trial = []
        for z in order[: self.tries]:
            if self.step[z] < 1.0:
                continue
            self.trial.append((z, self.nb.shift[z]))
            self.nb.shift[z] += self.step[z]

    def judge(self, lap: dict) -> None:
        clean = lap["status"] == "finished"
        tried = {z for z, _ in self.trial}
        for z, before in self.trial:
            t = self.zone_time(lap, z)
            if clean and t < self.best[z] - 0.5:
                self.best[z] = t
                self.kept += 1
            else:
                self.nb.shift[z] = before
                self.step[z] *= 0.5
        if clean:
            for z in set(range(len(self.nb.points))) - tried:
                self.best[z] = min(self.best[z], self.zone_time(lap, z))
        else:
            z = self.nb.zone_of(lap["point"] % self.nb.n)
            if z not in tried:
                self.nb.shift[z] -= self.crash_m  # crashed where I changed nothing: brake earlier there


def learn(args: tuple) -> dict:
    spec, laps_n, tries, step, crash_m, seed = args
    setup_d = json.load(open(WINNERS / f"{GENERALS[0]}.json"))["setup"]
    sim = fevo.Simulator(general.RULES)
    setup = [setup_d[k["id"]] for k in sim.rules.setup]
    sim.set_setup(setup)
    tr = general._track(spec)
    sim.load_track(tr)
    n = tr.points
    ex = LineExpert(tr, sim.rules, setup, offsets=racing_line(tr, -0.45), **EXPERT)
    obs, st = sim.reset(1), "running"
    ex.reset()
    while st == "running":
        a = ex(sim, obs)
        st = sim.step(a[0], a[1])
        obs = sim.observation()
    expert = sim.steps if st == "finished" else None
    best = max(GENERALS, key=lambda g: rank(official(sim, json.load(open(WINNERS / f"{g}.json")))[0]))
    sim.load_program(fevo.from_json(open(WINNERS / f"{best}.json").read()))
    starts = []
    for s in range(3):
        sim.reset(s)
        r = sim.run_agent_lap()
        starts.append((r == "finished", sim.progress_index, s))
    start = max(starts)[2]
    rng = np.random.default_rng(seed)
    lap1 = run_lap(sim, None, start)
    nb = Notebook(n, zones_from(lap1["brake_at"], n) or [0])
    pr = Practice(nb, tries, step, crash_m, rng)
    curve = [lap1["steps"] if lap1["status"] == "finished" else -round(lap1["point"] / n, 3)]
    if lap1["status"] == "finished":
        pr.remember(lap1)
    for _ in range(laps_n - 1):
        pr.plan()
        lap = run_lap(sim, nb, start)
        curve.append(lap["steps"] if lap["status"] == "finished" else -round(lap["point"] / n, 3))
        if not np.isfinite(pr.best).any() and lap["status"] == "finished":
            pr.remember(lap)
        else:
            pr.judge(lap)
    finals = [run_lap(sim, nb, s) for s in range(3)]
    notes = [f"{p / 1000:4.2f} km: brake {abs(s):.0f} m {'later' if s > 0 else 'earlier'}"
             for p, s in zip(nb.points, nb.shift) if abs(s) >= 1]
    return {"track": spec, "driver": best, "start": start, "zones": len(nb.points), "kept": pr.kept,
            "expert": expert, "curve": curve, "official": [(f["status"], f["steps"]) for f in finals],
            "notes": notes, "notebook": {"points": nb.points, "shift_m": [round(float(s), 1) for s in nb.shift]}}


def main() -> None:
    ap = argparse.ArgumentParser(prog="notebook", description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", default=",".join([f"gen:{s}" for s in general._distinct_gen(950_000, 5)] + ["spielberg", "montreal"]))
    ap.add_argument("--laps", type=int, default=30)
    ap.add_argument("--tries", type=int, default=2, help="zones tried per lap")
    ap.add_argument("--step", type=float, default=16.0, help="metres later tried at first")
    ap.add_argument("--crash-m", type=float, default=10.0)
    ap.add_argument("--out", default="notebook")
    ap.add_argument("-w", "--workers", type=int)
    a = ap.parse_args()
    specs = a.tracks.split(",")
    with mp.get_context("spawn").Pool(a.workers or min(len(specs), max(1, (os.cpu_count() or 2) - 1))) as pool:
        reps = pool.map(learn, [(s, a.laps, a.tries, a.step, a.crash_m, i) for i, s in enumerate(specs)])
    out = WINNERS / a.out
    out.mkdir(parents=True, exist_ok=True)
    for r in reps:
        name = r["track"].replace("gen:", "g")
        f = lambda c: fevo.format_lap_time(c) if c > 0 else f"x{-c:.0%}"  # noqa: E731
        print(f"\n{r['track']} (driver {r['driver']}, start {r['start']}, {r['zones']} braking zones, "
              f"{r['kept']} changes kept, expert {fevo.format_lap_time(r['expert']) if r['expert'] else 'fails'})")
        print("  laps: " + " ".join(f(c) for c in r["curve"]))
        print("  official starts after: " + " / ".join(fevo.format_lap_time(s) if st == "finished" else st for st, s in r["official"]))
        for line in r["notes"][:10]:
            print("   " + line)
        (out / f"{name}.json").write_text(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
