"""Score general drivers on the held-out circuits exactly as general.py does (five laps each), under general.RULES.

    uv run python -m fevoracer.rescore general-solo4-es general-solo4-robust2-laps ...
    FEVO_RULES=solo-4 uv run python -m fevoracer.rescore ...      # the old physics, for comparison
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import fevo

from . import general, realtracks
from .general import HELDOUT_GEN, summarise
from .paths import WINNERS


def main() -> None:
    names = sys.argv[1:] or ["general-solo4-es", "general-solo4-robust2-laps", "general-solo4-mixd-r30", "general-solo4-slow"]
    held = [f"gen:{s}" for s in HELDOUT_GEN] + realtracks.names("test") + ["spielberg"]
    setup_d = json.loads((WINNERS / f"{names[0]}.json").read_text())["setup"]
    rules = fevo.load_rules(general.RULES)
    setup = [setup_d[k["id"]] for k in rules.setup]
    with mp.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1), initializer=general._init, initargs=(setup,)) as pool:
        base = pool.map(general._official, [(h, None, True) for h in held])
        expert = {r["track"]: r["expert"] for r in base}
        print(f"rules {general.RULES}", flush=True)
        for name in names:
            mlp = json.loads((WINNERS / f"{name}.json").read_text())
            ev = pool.map(general._official, [(h, mlp, False) for h in held])
            for e in ev:
                e["expert"] = expert[e["track"]]
            line, score, fin, total = summarise(ev)
            print(f"{name:<30} laps finished {fin}/{total}, completed {score:.1%}\n   {line}", flush=True)


if __name__ == "__main__":
    main()
