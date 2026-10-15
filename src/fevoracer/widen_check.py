"""Widen a general driver to a later rules version's inputs (zero weights on the new ones) and prove it drives
identically: lap hashes on every shipped circuit, and the held-out laps general.py scores, old rules vs new.

    uv run python -m fevoracer.widen_check general-solo4-es general-solo6-w0 solo-5 solo-6
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import fevo

from . import general, realtracks
from .explore import widen
from .general import HELDOUT_GEN
from .paths import WINNERS


def _held(rules: str, mlp: dict, setup: list[float]) -> list[dict]:
    os.environ["FEVO_RULES"] = rules
    general.RULES = rules
    held = [f"gen:{s}" for s in HELDOUT_GEN] + realtracks.names("test") + ["spielberg"]
    with mp.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1), initializer=general._init, initargs=(setup,)) as pool:
        return pool.map(general._official, [(h, mlp, False) for h in held])


def main() -> None:
    src, dst, old, new = sys.argv[1:5]
    mlp = json.loads((WINNERS / f"{src}.json").read_text())
    wide = widen(mlp, len(fevo.load_rules(new).observation))
    (WINNERS / f"{dst}.json").write_text(json.dumps(wide))
    same = diff = 0
    for t in fevo.track_ids():
        a = fevo.evaluate(fevo.from_json(json.dumps(mlp)), t, old)
        b = fevo.evaluate(fevo.from_json(json.dumps(wide)), t, new)
        ha = [(l.status, l.steps, l.hash) for l in a.laps]
        hb = [(l.status, l.steps, l.hash) for l in b.laps]
        same, diff = same + (ha == hb), diff + (ha != hb)
        if ha != hb:
            print(f"  {t}: {old} {ha} vs {new} {hb}")
    print(f"shipped circuits: {same} identical, {diff} different (lap hashes)", flush=True)
    sd = mlp["setup"]
    setup = [sd[k["id"]] for k in fevo.load_rules(new).setup]
    ra, rb = _held(old, mlp, setup), _held(new, wide, setup)
    bad = [(x["track"], x["laps"], y["laps"]) for x, y in zip(ra, rb) if x["laps"] != y["laps"]]
    for t, x, y in bad:
        print(f"  {t}: {x} vs {y}")
    print(f"held-out circuits: {len(ra) - len(bad)} identical, {len(bad)} different (status, steps, progress)")
    print(general.summarise([dict(r, expert=[]) for r in rb])[1:])


if __name__ == "__main__":
    main()
