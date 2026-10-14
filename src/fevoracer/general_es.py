"""Stage 2 for the general driver: an evolution strategy across many circuits, after imitation has plateaued.

Imitation (general.py) copies an expert, and near the moment braking starts the expert's labels switch from full
throttle to hard braking within a few metres: a network fitted to them averages the two and coasts into the zone.
This optimises what matters directly, getting round, the way explore.py does for one circuit:

* every generation a fresh batch of circuits (generated, slow-corner and real training ones, never the held-out
  ones), so the search cannot fit one track;
* each candidate drives, on every circuit of the batch, a lap from an official start and short stints starting
  just before braking zones at the expert's speed (slow corners three times as likely), all candidates the same
  starts;
* fitness is metres of progress per second, a crash adding `crash_s` seconds, averaged over everything;
* antithetic pairs, rank shaping and an Adam step along the estimated gradient (OpenAI-ES); the masked inputs
  (lap progress) keep zero weights;
* every `eval_every` generations the mean network is scored on the held-out circuits exactly as general.py scores
  them (five laps each), and written when it beats the best so far.

    uv run python -m fevoracer.general_es --init winners/general-solo4-slow.json --name general-solo4-es
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

import fevo

from . import general, realtracks
from .general import EXCLUDE_REAL, HELDOUT_GEN, MASKED, braking_points, summarise
from .line import racing_line
from .paths import CHECKPOINTS, WINNERS
from .racer import LineExpert
from .smooth import flatten, unflatten as _unflatten


def unflatten(base: dict, x: np.ndarray) -> dict:
    """The network for parameters x, rounded to 6 decimals as general.py writes them (full precision would take
    the file past the format's 500 KB limit)."""
    return _unflatten(base, np.round(x, 6))


def _prep(spec: str) -> tuple[str, list[int], np.ndarray]:
    """A circuit's braking points and the expert's planned speed (for braking-zone starts)."""
    tr = general._track(spec)
    ex = LineExpert(tr, general._SIM.rules, general._SETUP, offsets=racing_line(tr, -0.45), **general.EXPERT)
    return spec, braking_points(ex.v), np.asarray(ex.v, dtype=np.float32)


_TRACKS: dict[str, fevo.Track] = {}


def _episodes(args: tuple) -> list[float]:
    """Fitness of a chunk of candidates on one circuit: mean progress rate over its lap and stints."""
    spec, base, cands, seed, starts, stint, crash_s = args
    sim = general._SIM
    if spec not in _TRACKS:
        _TRACKS.clear()
        _TRACKS[spec] = general._track(spec)
    tr = _TRACKS[spec]
    sim.load_track(tr)
    hz, n = sim.rules.agent_hz, tr.points
    out = []
    for x in cands:
        sim.load_program(fevo.from_json(json.dumps(unflatten(base, x))))
        rates = []
        sim.reset(seed)
        st = sim.run_agent_lap()
        dist = n if st == "finished" else max(0, sim.progress_index if sim.progress_index < n - 100 else 0)
        rates.append(dist / (sim.steps / hz + (0 if st == "finished" else crash_s)))
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
            t = sim.steps / hz + (crash_s if st not in ("running", "finished") else 0)
            rates.append(max(0, dist) / max(t, 1e-3))
        out.append(float(np.mean(rates)))
    return out


def run(init: str, name: str, generations: int, pairs: int, sigma: float, lr: float, gen_tracks: int, slow_tracks: int,
        real_tracks: int, stints: int, stint: int, crash_s: float, eval_every: int, chunk: int, workers: int | None,
        seed: int, fresh: bool, mix_tracks: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base = json.load(open(init))
    setup_d = base.get("setup")
    rules = fevo.load_rules(general.RULES)
    setup = [setup_d[k["id"]] for k in rules.setup] if setup_d else json.load(open("winners/solo3-spielberg.json"))["setup"]
    x = flatten(base)
    n_in, n_h = base["layers"][0], base["layers"][1]
    frozen = np.zeros(len(x), bool)
    for j in range(n_h):
        for i in MASKED:
            frozen[j * n_in + i] = True
    assert np.all(x[frozen] == 0), "the masked inputs must start with zero weights"
    real_train = [r for r in realtracks.names("train") if r not in EXCLUDE_REAL]
    held = [f"gen:{s}" for s in HELDOUT_GEN] + realtracks.names("test") + ["spielberg"]
    out, state_file = WINNERS / f"{name}.json", CHECKPOINTS / f"{name}-es.pkl"
    m, v, gen0, best = np.zeros_like(x), np.zeros_like(x), 0, -1.0
    if state_file.is_file() and not fresh:
        st = pickle.loads(state_file.read_bytes())
        x, m, v, gen0, best, rng = st["x"], st["m"], st["v"], st["gen"], st["best"], st["rng"]
        print(f"resumed at generation {gen0}, best held-out completion {best:.1%}", flush=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"General ES on {general.RULES}: {len(x)} numbers ({frozen.sum()} frozen), {2 * pairs} candidates a generation, "
          f"sigma {sigma}, lr {lr}; each generation {gen_tracks} generated + {slow_tracks} slow-corner + {mix_tracks} mixed + {real_tracks} real "
          f"training circuits, a lap and {stints} braking-zone stints of {stint} steps on each, crash +{crash_s} s; "
          f"held-out scoring every {eval_every} generations", flush=True)
    t0 = time.time()
    b1, b2, eps = 0.9, 0.999, 1e-8
    with mp.get_context("spawn").Pool(workers, initializer=general._init, initargs=(setup,)) as pool:

        def held_out(params: np.ndarray, label: str) -> None:
            nonlocal best
            mlp = unflatten(base, params)
            ev = pool.map(general._official, [(h, mlp, False) for h in held])
            for e in ev:
                e["expert"] = []
            line, score, fin, total = summarise(ev)
            mark = ""
            if score > best:
                best = score
                out.write_text(json.dumps(mlp) + "\n")
                mark = "  << new best"
            print(f"   held out ({label}): {line}  | completed {score:.1%}, laps finished {fin}/{total}{mark}", flush=True)

        if gen0 == 0 and eval_every > 0:
            held_out(x, "start")
        for gen in range(gen0 + 1, gen0 + generations + 1):
            specs = ([f"gen:{int(s)}" for s in rng.integers(1, 800_000, gen_tracks)]
                     + [f"slow:{int(s)}" for s in rng.integers(1, 800_000, slow_tracks)]
                     + [f"mix:{int(s)}" for s in rng.integers(1, 800_000, mix_tracks)]
                     + list(rng.choice(real_train, real_tracks, replace=False)))
            tg = time.time()
            prep = pool.map(_prep, specs)
            t_prep = time.time() - tg
            tasks = []
            eps_ = rng.standard_normal((pairs, len(x)))
            eps_[:, frozen] = 0.0
            cands = [x + sigma * e for e in eps_] + [x - sigma * e for e in eps_]
            for spec, bps, vplan in prep:
                n = len(vplan)
                starts = []
                if bps:
                    w = np.array([3.0 if vplan[(b + np.arange(400)) % n].min() * 3.6 <= 90 else 1.0 for b in bps])
                    for _ in range(stints):
                        p = (int(rng.choice(bps, p=w / w.sum())) - int(rng.integers(0, 200))) % n
                        starts.append((p, float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-0.02, 0.02)),
                                       float(rng.uniform(0.95, 1.05) * vplan[p])))
                lap_seed = int(rng.integers(3))
                for c0 in range(0, len(cands), chunk):
                    tasks.append((spec, base, cands[c0:c0 + chunk], lap_seed, starts, stint, crash_s))
            tg = time.time()
            res = pool.map(_episodes, tasks)
            t_eps = time.time() - tg
            per_track = len(cands) // chunk + (1 if len(cands) % chunk else 0)
            f = np.zeros(len(cands))
            for ti in range(len(prep)):
                fit = np.concatenate(res[ti * per_track:(ti + 1) * per_track])
                # Rank within each circuit, so a fast circuit does not outweigh a slow one.
                r = np.empty(len(fit))
                r[np.argsort(fit)] = np.arange(len(fit))
                f += r / (len(fit) - 1) - 0.5
            f /= len(prep)
            raw = np.mean([np.concatenate(res[ti * per_track:(ti + 1) * per_track]) for ti in range(len(prep))], 0)
            ranks = np.empty(len(f))
            ranks[np.argsort(f)] = np.arange(len(f))
            u = ranks / (len(f) - 1) - 0.5
            g = ((u[:pairs] - u[pairs:])[:, None] * eps_).sum(0) / (2 * pairs * sigma)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            step = lr * (m / (1 - b1 ** gen)) / (np.sqrt(v / (1 - b2 ** gen)) + eps)
            step[frozen] = 0.0
            x = x + step
            print(f"gen {gen:4d}  {(time.time() - t0) / 60:6.1f} min  progress rate mean {raw.mean():6.2f} max {raw.max():6.2f} m/s "
                  f"over {len(prep)} circuits  (prep {t_prep:.0f} s, driving {t_eps:.0f} s)", flush=True)
            if eval_every > 0 and gen % eval_every == 0:
                held_out(x, f"gen {gen}")
                state_file.write_bytes(pickle.dumps({"x": x, "m": m, "v": v, "gen": gen, "best": best, "rng": rng}))
    print(f"\nBest held-out completion {best:.1%}: {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="general_es", description=__doc__.split("\n")[0])
    ap.add_argument("--init", required=True, help="the general driver to start from")
    ap.add_argument("--name", default="general-es", help="writes winners/<name>.json")
    ap.add_argument("-g", "--generations", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=3e-3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gen-tracks", type=int, default=3)
    ap.add_argument("--slow-tracks", type=int, default=3)
    ap.add_argument("--real-tracks", type=int, default=3)
    ap.add_argument("--mix-tracks", type=int, default=0, help="circuits with late chicanes, fast kinks and compound corners")
    ap.add_argument("--stints", type=int, default=4, help="braking-zone stints per circuit")
    ap.add_argument("--stint", type=int, default=400)
    ap.add_argument("--crash-s", type=float, default=5.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=8, help="candidates per task")
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    run(a.init, a.name, a.generations, a.pairs, a.sigma, a.lr, a.gen_tracks, a.slow_tracks, a.real_tracks, a.stints,
        a.stint, a.crash_s, a.eval_every, a.chunk, a.workers, a.seed, a.fresh, a.mix_tracks)


if __name__ == "__main__":
    main()
