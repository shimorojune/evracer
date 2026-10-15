"""Lap hashes of every top-level winner on the shipped circuits, per rules version: proof that a simulator change
leaves earlier rules versions bit-identical.

    uv run python -m fevoracer.hashes logs/hashes-pre.json      # before the change
    uv run python -m fevoracer.hashes logs/hashes-post.json     # after; then compare the two files
"""

from __future__ import annotations

import json
import sys

import fevo

from .paths import WINNERS

RULES = ("solo-1", "solo-2", "solo-3", "solo-4")


def main() -> None:
    out = {}
    for f in sorted(WINNERS.glob("*.json")):
        try:
            p = fevo.from_json(f.read_text())
        except Exception:
            continue
        for rid in RULES:
            for t in fevo.track_ids():
                try:
                    r = fevo.evaluate(p, t, rid)
                except Exception:
                    continue  # input count does not fit this rules version
                out[f"{f.name}@{t}@{rid}"] = [(l.status, l.steps, l.hash) for l in r.laps]
    json.dump(out, open(sys.argv[1], "w"), indent=0, sort_keys=True)
    by = {rid: sum(1 for k in out if k.endswith(rid)) for rid in RULES}
    print(f"{len(out)} cases: {by}")


if __name__ == "__main__":
    main()
