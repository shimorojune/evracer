"""Practice session: find the best car setup for a circuit.

Like an F1 practice session: the same driver goes out with a bit less wing,
a bit more, a different balance, and the engineers keep what's quickest.
The driver is the setup-aware PPO network, so trying a setup costs three
official laps (about 0.3 s) and no retraining. CMA-ES chooses what to try
next. The winner is written as an upload file with its setup block.
"""

from __future__ import annotations

import json
import time
import warnings

import torch

import fevo

from .paths import CHECKPOINTS, winner_path
from .ppo import KNOBS, Policy, default_setup, mlp_to_neat, setup_dict, setup_from_norm, setup_norm

warnings.filterwarnings("ignore", message="Could not import matplotlib")
import cma  # noqa: E402

CRASH_PENALTY_S = 30.0


def _load_policy(track: str) -> Policy:
    for name in ("ppo-best.pt", "ppo.pt"):
        f = CHECKPOINTS / track / name
        if f.is_file():
            state = torch.load(f, weights_only=False)
            p = Policy()
            p.load_compatible(state["policy"])
            return p
    raise SystemExit(f"No PPO driver for {track}; run train-ppo first.")


def lap_score(sim: fevo.Simulator, policy: Policy, track: str, setup: list[float]) -> tuple[float, fevo.OfficialResult]:
    """Official median lap in seconds; unranked setups get the gates they reached plus a penalty."""
    res = fevo.evaluate(fevo.from_json(json.dumps(policy.to_mlp(setup))), track, sim=sim)
    if res.ranked:
        return res.median_steps / 60, res
    g = sorted(l.gates_passed for l in res.laps)[1]
    return 1000.0 + CRASH_PENALTY_S * (1 - g / max(1, res.laps[0].gate_count)), res


def practice(track: str, runs: int, seed: int) -> None:
    policy = _load_policy(track)
    sim = fevo.Simulator()
    t0 = time.time()
    base_s, base = lap_score(sim, policy, track, default_setup())
    print(f"Practice at {fevo.load_track(track).name}. Default setup: {base.lap_time or 'no lap'}", flush=True)
    es = cma.CMAEvolutionStrategy([0.0] * len(KNOBS), 0.4, {"bounds": [-1, 1], "popsize": 8, "seed": seed, "verbose": -9})
    best_s, best_setup, best_res = base_s, default_setup(), base
    tried = 1
    while tried < runs:
        xs = es.ask()
        scores = []
        for x in xs:
            setup = [round(v, 3) for v in setup_from_norm(list(x))]
            s, res = lap_score(sim, policy, track, setup)
            scores.append(s)
            tried += 1
            if s < best_s:
                best_s, best_setup, best_res = s, setup, res
                print(f"  run {tried:4d}: {res.lap_time}  {setup_dict(setup)}", flush=True)
        es.tell(xs, scores)
    mlp = policy.to_mlp(best_setup)
    winner_path("practice", track).write_text(json.dumps(mlp) + "\n")
    winner_path("practice", track, ".neat.json").write_text(json.dumps(mlp_to_neat(mlp)) + "\n")
    print(f"\n{tried} setups in {time.time() - t0:.0f} s. Default {base.lap_time or 'no lap'} -> best {best_res.lap_time or 'no lap'}")
    for k, v in zip(KNOBS, best_setup):
        print(f"  {k['label']:<24} {v:>6}   (default {k['default']}, range {k['min']} to {k['max']})")
    print(f"Upload file: {winner_path('practice', track)}")
