"""Replay laps exported from notebook-lab.html ("Export laps") exactly in Python, and check they match the page.

    uv run python -m fevoracer.notebook_replay notebook-laps-....json            # every lap
    uv run python -m fevoracer.notebook_replay notebook-laps-....json --lap 12   # one lap, with a trace around T1
"""

from __future__ import annotations

import argparse
import json

import fevo

from . import general
from .notebook2 import Notebook, run_lap
from .paths import WINNERS
from .practice import GENERALS


def main() -> None:
    ap = argparse.ArgumentParser(prog="notebook_replay", description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--lap", type=int, help="replay only this lap number")
    a = ap.parse_args()
    data = json.load(open(a.file))
    setup_d = json.loads((WINNERS / f"{GENERALS[0]}.json").read_text())["setup"]
    sims: dict[str, fevo.Simulator] = {}
    for lap in data["laps"]:
        if a.lap is not None and lap["no"] != a.lap:
            continue
        rid = lap.get("rules", data.get("rules", general.RULES))  # each lap's own rules (the page can switch)
        if rid not in sims:
            sims[rid] = fevo.Simulator(rid)
        sim = sims[rid]
        setup = [setup_d[k["id"]] for k in sim.rules.setup]
        sim.set_setup(setup)
        order = sim.rules.raw["carParamOrder"]
        tr = general._track(lap["track"])
        sim.load_track(tr)
        sim.load_program(fevo.from_json((WINNERS / f"{lap['driver']}.json").read_text()))
        car = sim.rules.packed_car(setup)
        for k, v in lap["aero"].items():
            car[order.index(k)] = v
        sim.core.write_f64(car)
        sim.core.call("setCarParamsFromScratch", len(car))
        nb = None
        if lap["notesOn"]:
            nb = Notebook(tr.points, lap["corners"]["points"], lap["corners"]["apex"], lap["corners"]["power"], edge_keep=0.2)
            for kind, vals in lap["notes"].items():
                nb.notes[kind][:] = vals
        r = run_lap(sim, nb, lap["start"])
        same = r["status"] == lap["status"] and r["steps"] == lap["steps"]
        fmt = lambda st, n: fevo.format_lap_time(n) if st == "finished" else f"{st} at step {n}"  # noqa: E731
        print(f"lap {lap['no']:3d} {rid} {lap['track']:<12} start {lap['start']} notes {'on' if lap['notesOn'] else 'off'}"
              f"{' trying ' + lap['trial']['kind'] + ' at corner ' + str(lap['trial']['z']) if lap.get('trial') else ''}: "
              f"page {fmt(lap['status'], lap['steps'])}, python {fmt(r['status'], r['steps'])}  {'identical' if same else 'DIFFERS'}")


if __name__ == "__main__":
    main()
