"""Notebook learning after a car change (Rohit's report): learn at 50% drag, then restore 100%; the committed rules vs the working copy.

    uv run python -m fevoracer.notebook_drag_check gen:950005 24
"""
import json, subprocess, sys, types, numpy as np, fevo
from fevoracer import general, notebook2 as new
from fevoracer.paths import WINNERS

src = subprocess.run(["git", "show", "HEAD:src/fevoracer/notebook2.py"], capture_output=True, text=True).stdout
src = src.replace("from . import general", "from fevoracer import general").replace("from .", "from fevoracer.")
old = types.ModuleType("nb2_old"); exec(src, old.__dict__)

spec, laps_each = sys.argv[1], int(sys.argv[2])
setup_d = json.load(open(WINNERS / "general-solo4-es.json"))["setup"]
sim = fevo.Simulator("solo-4"); setup = [setup_d[k["id"]] for k in sim.rules.setup]; sim.set_setup(setup)
order = sim.rules.raw["carParamOrder"]; car = sim.rules.packed_car(setup)
def drag(f):
    c = list(car); c[order.index("dragAreaM2")] *= f
    sim.core.write_f64(c); sim.core.call("setCarParamsFromScratch", len(c))
tr = general._track(spec); sim.load_track(tr); n = tr.points
sim.load_program(fevo.from_json((WINNERS / "general-solo4-es.json").read_text()))
for name, mod in (("old", old), ("fixed", new)):
    drag(1.0)
    lap1 = mod.run_lap(sim, None, 2)
    nb = mod.Notebook(n, *mod.corners(lap1, n), edge_keep=0.2)
    pr = mod.Practice(nb, np.random.default_rng(1), 3.0)
    print(f"\n== {name} rules on {spec}: corners at {[round(p / 1000, 2) for p in nb.points]} km; lap 1 (100% drag) {fevo.format_lap_time(lap1['steps']) if lap1['status'] == 'finished' else lap1['status']}")
    for phase, f in (("50% drag", 0.5), ("100% drag", 1.0)):
        drag(f)
        if f == 1.0 and hasattr(pr, "reprobe"):
            pr.reprobe()
        rows = []
        for i in range(laps_each):
            pr.plan()
            lap = mod.run_lap(sim, nb, 2)
            pr.judge(lap)
            res = fevo.format_lap_time(lap["steps"]) if lap["status"] == "finished" else f"{lap['status']}@{lap['point'] / n:.0%}"
            rows.append(f"{res}  brake m {[round(float(b)) for b in nb.notes['brake']]}")
        print(f"  {phase}:")
        for i, r in enumerate(rows):
            if i % 3 == 2 or i == len(rows) - 1:
                print(f"    lap {i + 1:2d}: {r}")
