"""How brittle a general driver is: held-out laps finished when its applied actions carry a slowly drifting error
(the same kind ES and DAgger practise with), at a few levels and noise seeds, under general.RULES.

    FEVO_RULES=solo-6 uv run python -m fevoracer.robust_check general-solo6-es general-solo6
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import fevo
import numpy as np

from . import general, realtracks
from .general import EXTRA_STARTS, HELDOUT_GEN
from .paths import WINNERS

LEVELS = (0.0, 0.03, 0.06, 0.1)
SEEDS = 3


def _laps(args: tuple) -> list[tuple[float, int, bool, float]]:
    spec, mlp, levels, seeds = args
    tr, _, _ = general._expert_for(spec)
    sim = general._SIM
    sim.load_track(tr)
    sim.load_program(fevo.from_json(json.dumps(mlp)))
    out = []
    for lvl in levels:
        for k in range(seeds if lvl else 1):
            rng = np.random.default_rng([k, hash(spec) & 0xFFFF])
            for s in (0, 1, 2, *EXTRA_STARTS):
                obs = sim.reset(s) if isinstance(s, int) else sim.reset(1, lateral_m=s[0], heading_rad=s[1])
                drift, st = np.zeros(2), "running"
                while st == "running":
                    a = sim.evaluate_program(obs)
                    drift = 0.9 * drift + rng.normal(0.0, lvl * 0.436, 2)
                    st = sim.step(float(np.clip(a[0] + drift[0], -1, 1)), float(np.clip(a[1] + drift[1], -1, 1)))
                    obs = sim.observation()
                out.append((lvl, k, st == "finished", 1.0 if st == "finished" else sim.progress_index / tr.points))
    return out


def main() -> None:
    names = sys.argv[1:]
    held = [f"gen:{s}" for s in HELDOUT_GEN] + realtracks.names("test") + ["spielberg"]
    sd = json.loads((WINNERS / f"{names[0]}.json").read_text())["setup"]
    setup = [sd[k["id"]] for k in fevo.load_rules(general.RULES).setup]
    print(f"rules {general.RULES}; drifting action error, {SEEDS} noise seeds per level; laps finished of 65 per seed, mean completion", flush=True)
    with mp.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1), initializer=general._init, initargs=(setup,)) as pool:
        for name in names:
            mlp = json.loads((WINNERS / f"{name}.json").read_text())
            res = sum(pool.map(_laps, [(h, mlp, LEVELS, SEEDS) for h in held]), [])
            cells = []
            for lvl in LEVELS:
                r = [x for x in res if x[0] == lvl]
                n = len({x[1] for x in r})
                cells.append(f"noise {lvl:.2f}: {sum(x[2] for x in r) / n:4.1f}/65 {np.mean([x[3] for x in r]):.0%}")
            print(f"{name:<24} " + "   ".join(cells), flush=True)


if __name__ == "__main__":
    main()
