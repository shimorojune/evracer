"""Notebook v2: learning a track by notes per corner (braking point, line, exit throttle), driver weights frozen.

The general driver steers and drives; a per-track notebook (Python side only, keyed by position on the lap)
holds three notes per corner, each applied as a nudge the driver can still overrule, never as a takeover:

* braking point, metres later (+) or earlier (-) than the driver's own instinct (as v1: the throttle stays open
  until the marker; braking earlier brakes from the new marker);
* line: a small steering bias from 80 m before the braking point to the apex (the slowest point of the corner on
  the driver's own lap), so it turns in from further left or right. It is only applied while the tyres on that
  side keep more than `edge_keep` of their margin to the edge;
* exit: metres earlier that the driver gets back on the power after the apex. Applied as a floor of 0.3 on the
  pedal, and only while neither axle is sliding: the moment the rears or fronts let go it is the driver's pedal
  again (v1's lesson: forcing throttle on a driver at its grip limit breaks it).

Practice, from the driver's own laps only (no expert, no racing line): lap 1 is the driver alone and finds the
corners (where it starts braking, its apex, where it is back on the power). Each lap tries one note at one corner,
in small steps (3 m, 0.02 of steering lock, 5 m); a change is kept if the lap is clean and that corner's stretch
got quicker, otherwise it is put back and that note's step is halved. When every step is spent the steps are
refilled at half size, so practice keeps probing other corners instead of stopping. A crash at a corner whose
notes were not changed that lap moves its braking point 3 m earlier; stopping short of a corner (too slow to
carry on) moves it 3 m later. While no lap is clean, a change that gets the car further round counts as better, and
braking and line notes are tried both ways. reprobe() starts the tries again after the car changes.

    uv run python -m fevoracer.notebook2
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
from .notebook import LEAD, zones_from
from .paths import WINNERS
from .practice import GENERALS, official, rank
from .racer import LineExpert

REAR_SLIDE, FRONT_SLIDE, MARGIN_L, MARGIN_R = 18, 19, 21, 22
KINDS = {"brake": 3.0, "line": 0.02, "exit": 5.0}
EASE_EVERY = 4  # every 4th lap eases one note back towards the driver's own instinct
MIN_STEP = {"brake": 0.5, "line": 0.004, "exit": 1.0}


class Notebook:
    def __init__(self, n: int, points: list[int], apex: list[int], power: list[int], edge_keep: float):
        self.n, self.points, self.apex, self.power, self.edge_keep = n, points, apex, power, edge_keep
        k = len(points)
        self.notes = {kind: np.zeros(k) for kind in KINDS}

    def _d(self, j: int, p: int) -> int:
        return (j - p + self.n // 2) % self.n - self.n // 2

    def zone_of(self, j: int) -> int:
        k = len(self.points)
        for i, p in enumerate(self.points):
            span = (self.points[(i + 1) % k] - p) % self.n or self.n
            if (j - (p - LEAD)) % self.n < span:
                return i
        return k - 1

    def apply(self, j: int, steer: float, own: float, obs) -> tuple[float, float]:
        thr = own
        for i, p in enumerate(self.points):
            d = self._d(j, p)
            s = self.notes["brake"][i]
            if s > 0 and -60 <= d < s and own < 0:
                thr = 1.0
            elif s < 0 and s <= d < 0:
                thr = min(own, -0.7)
            bias = self.notes["line"][i]
            to_apex = self._d(self.apex[i], p)
            if bias and -80 <= d < max(to_apex, 1):
                # Negative steering is to the left: only lean that way while the tyres there have margin to spare.
                margin = obs[MARGIN_L] if bias < 0 else obs[MARGIN_R]
                if margin > self.edge_keep:
                    steer += bias
            e = self.notes["exit"][i]
            dp = self._d(j, self.power[i])
            if e > 0 and -e <= dp < 0 and own < 0.3 and max(abs(obs[REAR_SLIDE]), abs(obs[FRONT_SLIDE])) < 0.05:
                thr = 0.3
        return steer, thr

    def readable(self) -> list[str]:
        out = []
        for i, p in enumerate(self.points):
            parts = []
            b, ln, e = self.notes["brake"][i], self.notes["line"][i], self.notes["exit"][i]
            if abs(b) >= 0.5:
                parts.append(f"brake {abs(b):.0f} m {'later' if b > 0 else 'earlier'}")
            if abs(ln) >= 0.004:
                parts.append(f"turn in from further {'left' if ln < 0 else 'right'} ({abs(ln):.2f} lock)")
            if e >= 1:
                parts.append(f"back on the power {e:.0f} m earlier")
            if parts:
                out.append(f"{p / 1000:4.2f} km: " + ", ".join(parts))
        return out


def run_lap(sim: fevo.Simulator, nb: Notebook | None, seed: int) -> dict:
    n = sim.track.points
    obs, st = sim.reset(seed), "running"
    at, v, thr_own = np.zeros(n), np.full(n, np.inf), np.zeros(n)
    brake_at, braking = [], False
    while st == "running":
        j = sim.progress_index % n
        steer, own = sim.evaluate_program(obs)
        own = float(np.clip(own, -1, 1))
        steer, thr = nb.apply(j, float(steer), own, obs) if nb else (float(steer), own)
        if thr < -0.05 and not braking:
            brake_at.append(j)
        braking = thr < -0.05
        at[j] += 1
        v[j] = min(v[j], obs[11] * 100.0)
        thr_own[j] = own
        st = sim.step(float(np.clip(steer, -1, 1)), float(np.clip(thr, -1, 1)))
        obs = sim.observation()
    return {"status": st, "steps": sim.steps, "point": sim.progress_index, "at": at, "v": v, "thr": thr_own,
            "brake_at": brake_at}


def corners(lap: dict, n: int) -> tuple[list[int], list[int], list[int]]:
    """From the driver's own lap: braking points, the slowest point after each (apex), and where it is back on
    the power after that."""
    pts = zones_from(lap["brake_at"], n) or [0]
    apex, power = [], []
    for i, p in enumerate(pts):
        nxt = pts[(i + 1) % len(pts)]
        span = (nxt - p) % n or n
        idx = (p + np.arange(min(span, 500))) % n
        a = int(idx[int(np.argmin(np.where(np.isfinite(lap["v"][idx]), lap["v"][idx], np.inf)))])
        after = (a + np.arange(1, 300)) % n
        on = [j for j in after if lap["thr"][j] > 0.3]
        apex.append(a)
        power.append(int(on[0]) if on else int((a + 30) % n))
    return pts, apex, power


class Practice:
    def __init__(self, nb: Notebook, rng: np.random.Generator, crash_m: float):
        self.nb, self.rng, self.crash_m = nb, rng, crash_m
        k = len(nb.points)
        self.step = {kind: np.full(k, s) for kind, s in KINDS.items()}
        self.sign = {"line": self.rng.choice([-1.0, 1.0], k), "brake": np.ones(k), "exit": np.ones(k)}
        self.best = np.full(k, np.inf)
        self.reach = 0.0  # how far round the best unfinished lap since the last clean one got (1.0 once clean)
        self.laps = 0
        self.trial: tuple[str, int, float] | None = None
        self.kept = {kind: 0 for kind in KINDS}
        self.refills = 0

    def zone_time(self, lap: dict, z: int) -> float:
        nb = self.nb
        a = (nb.points[z] - LEAD) % nb.n
        span = (nb.points[(z + 1) % len(nb.points)] - nb.points[z]) % nb.n or nb.n
        return float(lap["at"][(a + np.arange(span)) % nb.n].sum())

    def remember(self, lap: dict) -> None:
        for z in range(len(self.nb.points)):
            self.best[z] = min(self.best[z], self.zone_time(lap, z))

    def plan(self) -> None:
        self.laps += 1
        noted = [(kind, z) for kind in KINDS for z in range(len(self.nb.points)) if abs(self.nb.notes[kind][z]) >= MIN_STEP[kind]]
        if noted and self.laps % EASE_EVERY == 0:
            # Every few laps, ease one note back towards the driver's own instinct; it stays eased unless that is worse,
            # so caution learned for other conditions can wear off.
            kind, z = noted[int(self.rng.integers(len(noted)))]
            before = self.nb.notes[kind][z]
            self.nb.notes[kind][z] = before - np.sign(before) * min(abs(before), max(KINDS[kind], 0.3 * abs(before)))
            self.trial = (kind, z, before, True)
            return
        cands = [(self.step[kind][z] / KINDS[kind], self.rng.random(), kind, z)
                 for kind in KINDS for z in range(len(self.nb.points)) if self.step[kind][z] >= MIN_STEP[kind]]
        if not cands:
            # Every note has been tried down to its smallest step: start again with half-size steps.
            self.refills += 1
            for kind in KINDS:
                self.step[kind][:] = KINDS[kind] / 2 ** self.refills
            return self.plan()
        _, _, kind, z = max(cands)
        before = self.nb.notes[kind][z]
        delta = self.step[kind][z] * self.sign[kind][z]
        self.nb.notes[kind][z] = before + delta
        self.trial = (kind, z, before, False)

    def reprobe(self) -> None:
        """Conditions changed (e.g. the car): forget what the old laps proved and try every note again, both ways."""
        for kind in KINDS:
            self.step[kind][:] = KINDS[kind]
        self.best[:] = np.inf
        self.reach = 0.0
        self.refills = 0

    def judge(self, lap: dict) -> None:
        clean = lap["status"] == "finished"
        reach = 1.0 if clean else min(0.999, lap["point"] / self.nb.n)
        if self.trial:
            kind, z, before, easing = self.trial
            t = self.zone_time(lap, z)
            # Better is a quicker clean lap through that corner, or, while no lap is clean, getting further round.
            # An eased note only has to be no worse: as quick (within 0.5 steps) when clean, as far round when not.
            if easing:
                better = (clean and t <= self.best[z] + 0.5) or (not clean and reach >= self.reach - 0.002)
            else:
                better = (clean and t < self.best[z] - 0.5) or (not clean and reach > self.reach + 0.002)
            if better:
                self.best[z] = min(self.best[z], t) if clean else self.best[z]
                self.kept[kind] += 1
            elif easing:
                self.nb.notes[kind][z] = before
            else:
                self.nb.notes[kind][z] = before
                self.step[kind][z] *= 0.5
                if kind != "exit":
                    self.sign[kind][z] *= -1  # that way did not help: try the other way next
        if clean:
            for z in range(len(self.nb.points)):
                if not self.trial or z != self.trial[1]:
                    self.best[z] = min(self.best[z], self.zone_time(lap, z))
        else:
            z = self.nb.zone_of(lap["point"] % self.nb.n)
            if not self.trial or z != self.trial[1]:
                if lap["status"] == "crashed":
                    self.nb.notes["brake"][z] -= self.crash_m  # went off: brake earlier there
                else:
                    # Stopped short of it: I am braking too early there. Undo a good share of any earlier braking at once.
                    b = self.nb.notes["brake"][z]
                    self.nb.notes["brake"][z] += max(self.crash_m, 0.3 * -b) if b < 0 else self.crash_m
        self.reach = 1.0 if clean else max(self.reach, reach)


def learn(args: tuple) -> dict:
    spec, laps_n, edge_keep, crash_m, seed = args
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
    lap1 = run_lap(sim, None, start)
    nb = Notebook(n, *corners(lap1, n), edge_keep=edge_keep)
    pr = Practice(nb, np.random.default_rng(seed), crash_m)
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
    crashes = sum(1 for c in curve[1:] if c <= 0)
    return {"track": spec, "driver": best, "start": start, "corners": len(nb.points), "kept": pr.kept,
            "crash_rate": round(crashes / max(1, len(curve) - 1), 3), "expert": expert, "curve": curve,
            "official": [(f["status"], f["steps"]) for f in finals], "notes": nb.readable(),
            "notebook": {"points": nb.points, "apex": nb.apex, "power": nb.power,
                         **{k: [round(float(x), 3) for x in v] for k, v in nb.notes.items()}}}


def main() -> None:
    ap = argparse.ArgumentParser(prog="notebook2", description=__doc__.split("\n")[0])
    gen = [f"gen:{s}" for s in general._distinct_gen(950_000, 5)] + [f"gen:{s}" for s in general._distinct_gen(960_000, 3)]
    ap.add_argument("--tracks", default=",".join(gen + ["spielberg", "montreal"]))
    ap.add_argument("--laps", type=int, default=30)
    ap.add_argument("--edge-keep", type=float, default=0.2, help="tyre margin (input units) kept on the side the line leans to")
    ap.add_argument("--crash-m", type=float, default=3.0)
    ap.add_argument("--out", default="notebook2")
    ap.add_argument("-w", "--workers", type=int)
    a = ap.parse_args()
    specs = a.tracks.split(",")
    with mp.get_context("spawn").Pool(a.workers or min(len(specs), max(1, (os.cpu_count() or 2) - 1))) as pool:
        reps = pool.map(learn, [(s, a.laps, a.edge_keep, a.crash_m, i) for i, s in enumerate(specs)])
    out = WINNERS / a.out
    out.mkdir(parents=True, exist_ok=True)
    for r in reps:
        name = r["track"].replace("gen:", "g")
        f = lambda c: fevo.format_lap_time(c) if c > 0 else f"x{-c:.0%}"  # noqa: E731
        print(f"\n{r['track']} (driver {r['driver']}, start {r['start']}, {r['corners']} corners, kept {r['kept']}, "
              f"crash rate {r['crash_rate']:.0%}, expert {fevo.format_lap_time(r['expert']) if r['expert'] else 'fails'})")
        print("  laps: " + " ".join(f(c) for c in r["curve"]))
        print("  official starts after: " + " / ".join(fevo.format_lap_time(s) if st == "finished" else st for st, s in r["official"]))
        for line in r["notes"][:10]:
            print("   " + line)
        (out / f"{name}.json").write_text(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
