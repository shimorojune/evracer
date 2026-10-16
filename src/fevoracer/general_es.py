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

from . import general, realtracks, robust_check
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


def _episodes(args: tuple) -> list[tuple[float, float]]:
    """Fitness of a chunk of candidates on one circuit (mean progress rate over its lap and stints) and the share of
    stints survived. With noise (act, obs, seed) the stints are driven with explore.py's mistakes: Gaussian noise on
    what the driver sees and a slowly drifting error on what it does (Ornstein-Uhlenbeck, about half a second), the
    same noise for every candidate. The lap from the official start is driven clean unless lap_noise is set, then
    with the same drifting action error."""
    spec, base, cands, seed, starts, stint, crash_s, noise = args
    act_n, obs_n, noise_seed = noise[:3]
    lap_noise = noise[3] if len(noise) > 3 else False
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
        # lap_noise 2: the lap twice, clean (pace) and with the action error (robustness), both counting.
        for noisy_lap in ((False, True) if lap_noise == 2 else (bool(lap_noise),)):
            obs = sim.reset(seed)
            if noisy_lap and act_n:
                rng = np.random.default_rng([noise_seed, 999])
                ou, st = np.zeros(2), "running"
                while st == "running":
                    steer, thr = sim.evaluate_program(obs)
                    ou += -ou / 30 + act_n * np.sqrt(2 / 30) * rng.standard_normal(2)
                    code, vals = sim.step_fast(float(np.clip(steer + ou[0], -1, 1)), float(np.clip(thr + ou[1], -1, 1)))
                    obs = list(vals[:len(obs)])
                    st = fevo.sim.STATUS[code]
            else:
                st = sim.run_agent_lap()
            dist = n if st == "finished" else max(0, sim.progress_index if sim.progress_index < n - 100 else 0)
            rates.append(dist / (sim.steps / hz + (0 if st == "finished" else crash_s)))
        survived = 0
        for k, (point, lat, hdg, speed) in enumerate(starts):
            obs = sim.reset(1, point=point, lateral_m=lat, heading_rad=hdg, speed=speed)
            prev, dist, st = point, 0, "running"
            rng = np.random.default_rng([noise_seed, k])
            ou = np.zeros(2)
            n_obs = len(obs)
            for _ in range(stint):
                if act_n == 0 and obs_n == 0:
                    st = sim.agent_step()
                else:
                    seen = list(np.asarray(obs) + obs_n * rng.standard_normal(n_obs)) if obs_n else obs
                    steer, thr = sim.evaluate_program(seen)
                    ou += -ou / 30 + act_n * np.sqrt(2 / 30) * rng.standard_normal(2)
                    code, vals = sim.step_fast(float(np.clip(steer + ou[0], -1, 1)), float(np.clip(thr + ou[1], -1, 1)))
                    obs = list(vals[:n_obs])
                    st = fevo.sim.STATUS[code]
                idx = sim.progress_index
                d = idx - prev
                dist += d + n if d < -n // 2 else d - n if d > n // 2 else d
                prev = idx
                if st != "running":
                    break
            survived += st in ("running", "finished")
            t = sim.steps / hz + (crash_s if st not in ("running", "finished") else 0)
            rates.append(max(0, dist) / max(t, 1e-3))
        out.append((float(np.mean(rates)), survived / max(1, len(starts))))
    return out


def run(init: str, name: str, generations: int, pairs: int, sigma: float, lr: float, gen_tracks: int, slow_tracks: int,
        real_tracks: int, stints: int, stint: int, crash_s: float, eval_every: int, chunk: int, workers: int | None,
        seed: int, fresh: bool, mix_tracks: int = 0, act_noise: float = 0.0, obs_noise: float = 0.0,
        noise_max: float = 0.1, noise_up: float = 1.2, survive: float = 0.8, lap_noise: int = 0,
        robust_select: float = 0.0, pace_select: bool = False) -> None:
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
    m, v, gen0, best = np.zeros_like(x), np.zeros_like(x), 0, (-1, -1.0)
    noise = {"act": act_noise, "obs": obs_noise}
    if state_file.is_file() and not fresh:
        st = pickle.loads(state_file.read_bytes())
        x, m, v, gen0, best, rng = st["x"], st["m"], st["v"], st["gen"], st["best"], st["rng"]
        noise = st.get("noise", noise)
        best = st.get("best_laps", (-1, best))
        print(f"resumed at generation {gen0}, best held-out {best[0]} laps, completion {best[1]:.1%}", flush=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"General ES on {general.RULES}: {len(x)} numbers ({frozen.sum()} frozen), {2 * pairs} candidates a generation, "
          f"sigma {sigma}, lr {lr}; each generation {gen_tracks} generated + {slow_tracks} slow-corner + {mix_tracks} mixed + {real_tracks} real "
          f"training circuits, a lap and {stints} braking-zone stints of {stint} steps on each, crash +{crash_s} s; "
          f"held-out scoring every {eval_every} generations; stint noise act {act_noise} obs {obs_noise} (x{noise_up} when "
          f"{survive:.0%} of stints survive, up to {noise_max}), the official-start lap {['clean', 'with the same action error', 'twice, clean and with the action error'][lap_noise]}; "
          f"held-out pick by {'clean laps + laps under action error ' + str(robust_select) if robust_select else 'clean laps'}", flush=True)
    t0 = time.time()
    b1, b2, eps = 0.9, 0.999, 1e-8
    with mp.get_context("spawn").Pool(workers, initializer=general._init, initargs=(setup,)) as pool:

        expert = {}
        floor = None
        if pace_select:
            expert = {r["track"]: r["expert"] for r in pool.map(general._official, [(h, None, True) for h in held])}

        def pace_of(ev: list[dict]) -> float:
            """Mean official-lap time against the expert over the held-out circuits (median of three); a circuit
            where any official lap fails counts as 30% slower than the expert."""
            g = []
            for e in ev:
                off = [s_ for st, s_, _ in e["laps"][:3] if st == "finished"]
                ex = sorted(s_ for st, s_ in e["expert"] if st == "finished")
                if len(ex) == 3:
                    g.append(sorted(off)[1] / ex[1] if len(off) == 3 else 1.3)
            return float(np.mean(g))

        def held_out(params: np.ndarray, label: str) -> None:
            nonlocal best, floor
            mlp = unflatten(base, params)
            ev = pool.map(general._official, [(h, mlp, False) for h in held])
            for e in ev:
                e["expert"] = expert.get(e["track"], [])
            line, score, fin, total = summarise(ev)
            mark, key = "", fin
            tie = score
            if robust_select:
                # Robustness counts as much as clean driving: the same laps again with a drifting action error.
                noisy = sum(pool.map(robust_check._laps, [(h, mlp, (robust_select,), 1) for h in held]), [])
                nfin = sum(r[2] for r in noisy)
                key = fin + nfin
                line += f"  | with error {robust_select}: {nfin}/{len(noisy)} ({np.mean([r[3] for r in noisy]):.0%})"
                if pace_select:
                    # Pace, but only while robustness holds: the starting network's clean and noisy laps are the
                    # floor (one lap of slack on clean, two on the single noisy run), the fastest network above it wins.
                    pace = pace_of(ev)
                    floor = floor or (fin - 1, nfin - 2)
                    ok = fin >= floor[0] and nfin >= floor[1]
                    key, tie = int(ok), -pace
                    line += f"  | pace {100 * (pace - 1):+.1f}% vs expert{'' if ok else ' (below the robustness floor)'}"
            # Every scored network is kept, so a good one that is not the pick is not lost.
            (CHECKPOINTS / f"{name}-{label.replace(' ', '')}.json").write_text(json.dumps(mlp) + "\n")
            # Laps finished on the held-out tracks first, mean completion as the tiebreak.
            if (key, tie) > best:
                best = (key, tie)
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
                    tasks.append((spec, base, cands[c0:c0 + chunk], lap_seed, starts, stint, crash_s,
                                  (noise["act"], noise["obs"], gen, lap_noise)))
            tg = time.time()
            res = pool.map(_episodes, tasks)
            t_eps = time.time() - tg
            per_track = len(cands) // chunk + (1 if len(cands) % chunk else 0)
            f = np.zeros(len(cands))
            res_f = [[fv for fv, _ in r] for r in res]
            surv = float(np.mean([sv for r in res for _, sv in r]))
            for ti in range(len(prep)):
                fit = np.concatenate(res_f[ti * per_track:(ti + 1) * per_track])
                # Rank within each circuit, so a fast circuit does not outweigh a slow one.
                r = np.empty(len(fit))
                r[np.argsort(fit)] = np.arange(len(fit))
                f += r / (len(fit) - 1) - 0.5
            f /= len(prep)
            raw = np.mean([np.concatenate(res_f[ti * per_track:(ti + 1) * per_track]) for ti in range(len(prep))], 0)
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
                  f"over {len(prep)} circuits, stints survived {surv:4.0%} at noise act {noise['act']:.3f} obs {noise['obs']:.3f}  "
                  f"(prep {t_prep:.0f} s, driving {t_eps:.0f} s)", flush=True)
            # Curriculum, as in explore.py: once most noisy stints survive, make the mistakes bigger.
            if surv >= survive and (noise["act"] or noise["obs"]):
                noise = {k: min(noise_max, v_ * noise_up) for k, v_ in noise.items()}
            if eval_every > 0 and gen % eval_every == 0:
                held_out(x, f"gen {gen}")
                state_file.write_bytes(pickle.dumps({"x": x, "m": m, "v": v, "gen": gen, "best": best[1], "best_laps": best, "rng": rng, "noise": noise}))
    print(f"\nBest held-out: {best[0]} laps, completion {best[1]:.1%}: {out}", flush=True)


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
    ap.add_argument("--act-noise", type=float, default=0.0, help="drifting action error in the stints (starting level)")
    ap.add_argument("--obs-noise", type=float, default=0.0, help="observation noise in the stints (starting level)")
    ap.add_argument("--noise-max", type=float, default=0.1)
    ap.add_argument("--noise-up", type=float, default=1.2, help="noise multiplier once enough stints survive")
    ap.add_argument("--survive", type=float, default=0.8, help="share of surviving stints that raises the noise")
    ap.add_argument("--lap-noise", action="store_true", help="drive the official-start lap with the stint action error too")
    ap.add_argument("--lap-both", action="store_true", help="drive the official-start lap twice: clean and with the stint action error")
    ap.add_argument("--robust-select", type=float, default=0.0, help="pick the held-out best by clean laps + laps under this action error")
    ap.add_argument("--pace-select", action="store_true", help="with --robust-select: pick the fastest network that keeps the start's robustness")
    ap.add_argument("--chunk", type=int, default=8, help="candidates per task")
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    run(a.init, a.name, a.generations, a.pairs, a.sigma, a.lr, a.gen_tracks, a.slow_tracks, a.real_tracks, a.stints,
        a.stint, a.crash_s, a.eval_every, a.chunk, a.workers, a.seed, a.fresh, a.mix_tracks,
        a.act_noise, a.obs_noise, a.noise_max, a.noise_up, a.survive, 2 if a.lap_both else int(a.lap_noise), a.robust_select, a.pace_select)


if __name__ == "__main__":
    main()
