"""Re-time laps on another build of the simulator (e.g. a prototype fix), against the shipped one.

Loads the alternative sim.wasm in place of the SDK's (this process only) and drives the same laps as
loophole_scan: the general drivers on every real circuit (three starts) and the saved notebooks (their start).

    uv run python -m fevoracer.retime --wasm path/to/sim-fix.wasm
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import fevo
import fevo._wasm as fw

from . import general, realtracks
from .notebook2 import Notebook, run_lap
from .paths import WINNERS


def laps(tag: str) -> dict:
    setup_d = json.loads((WINNERS / "general-solo4-es.json").read_text())["setup"]
    sim = fevo.Simulator(general.RULES)
    sim.set_setup([setup_d[k["id"]] for k in sim.rules.setup])
    out = {}
    for name in ("general-solo4-es", "general-solo4-robust2-laps", "general-solo4-mixd-r30"):
        sim.load_program(fevo.from_json((WINNERS / f"{name}.json").read_text()))
        for spec in realtracks.names("train") + realtracks.names("test"):
            sim.load_track(general._track(spec))
            for s in range(3):
                r = run_lap(sim, None, s)
                out[(name, spec, s)] = (r["status"], r["steps"])
    for folder in ("notebook", "notebook2"):
        for f in sorted((WINNERS / folder).glob("*.json")):
            r = json.loads(f.read_text())
            if "notebook" not in r:
                continue
            nbj = r["notebook"]
            k = len(nbj["points"])
            tr = general._track(r["track"])
            sim.load_track(tr)
            sim.load_program(fevo.from_json((WINNERS / f"{r['driver']}.json").read_text()))
            nb = Notebook(tr.points, nbj["points"], nbj.get("apex", nbj["points"]), nbj.get("power", nbj["points"]), 0.2)
            nb.notes["brake"][:] = nbj.get("brake", nbj.get("shift_m", [0.0] * k))
            nb.notes["line"][:] = nbj.get("line", [0.0] * k)
            nb.notes["exit"][:] = nbj.get("exit", [0.0] * k)
            lap = run_lap(sim, nb, r["start"])
            out[(folder, r["track"], r["start"])] = (lap["status"], lap["steps"])
    for folder in ("practice", "practice2", "practice3", "practice4", "distill-teachers"):
        d = WINNERS / folder
        if not (d / "report.json").exists():
            continue
        for r in json.loads((d / "report.json").read_text()):
            last = r["points"][-1]["budget"]
            f = d / f"{r['track'].replace('gen:', 'g')}-{last}.json"
            if not f.exists():
                continue
            sim.load_track(general._track(r["track"]))
            sim.load_program(fevo.from_json(f.read_text()))
            for s in range(3):
                lap = run_lap(sim, None, s)
                out[(f"{folder}-{last}", r["track"], s)] = (lap["status"], lap["steps"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="retime", description=__doc__.split("\n")[0])
    ap.add_argument("--wasm", required=True)
    a = ap.parse_args()
    base = laps("shipped")
    data = open(a.wasm, "rb").read()
    fw.wasm_bytes = lambda: data  # noqa: E731
    fw._engine_and_module.cache_clear()
    alt = laps("alternative")
    same = sum(base[k] == alt[k] for k in base)
    print(f"{len(base)} laps: {same} identical on both builds")
    rows = []
    for k in base:
        if base[k] != alt[k]:
            (sa, na), (sb, nb_) = base[k], alt[k]
            f = lambda st, n: fevo.format_lap_time(n) if st == "finished" else st  # noqa: E731
            d = f"{(nb_ - na) / 60:+.2f} s" if sa == sb == "finished" else ""
            rows.append(f"  {k[0]:<28} {k[1]:<14} start {k[2]}: {f(sa, na):>9} -> {f(sb, nb_):>9} {d}")
    print("\n".join(rows))
    fin_a = [base[k][1] for k in base if base[k][0] == "finished" and alt[k][0] == "finished"]
    fin_b = [alt[k][1] for k in base if base[k][0] == "finished" and alt[k][0] == "finished"]
    if fin_a:
        print(f"laps finished on both: {len(fin_a)}, mean change {(np.mean(fin_b) - np.mean(fin_a)) / 60:+.3f} s; "
              f"finished laps: shipped {sum(v[0] == 'finished' for v in base.values())}, alternative {sum(v[0] == 'finished' for v in alt.values())}")


if __name__ == "__main__":
    main()
