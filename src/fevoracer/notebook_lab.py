"""Build the notebook lab: the frozen general driver plus its learned track notes, live in the browser.

The page embeds the fevo SDK's exact sim.wasm (solo-4), every track that has a saved notebook (winners/notebook,
v1: braking points; winners/notebook2, v2: braking point, line and exit notes), and the general drivers they used.
It steps the simulator itself so the notes act live, exactly as notebook2.Notebook.apply does in Python: the page
shows the Python lap time with the same notes, start and stock aero next to its own.

    uv run python -m fevoracer.notebook_lab -o notebook-lab.html
"""

from __future__ import annotations

import argparse
import base64
import json
from importlib import resources
from pathlib import Path

import fevo
from fevo._wasm import wasm_bytes

from . import general
from .notebook2 import Notebook, run_lap
from .paths import WINNERS
from .practice import GENERALS

VERSIONS = (("notebook", 1), ("notebook2", 2))


def entries() -> list[dict]:
    out = []
    for folder, ver in VERSIONS:
        for f in sorted((WINNERS / folder).glob("*.json")):
            if f.name == "report.json":
                continue
            r = json.loads(f.read_text())
            nb = r["notebook"]
            k = len(nb["points"])
            notes = {"brake": nb.get("brake", nb.get("shift_m", [0.0] * k)), "line": nb.get("line", [0.0] * k),
                     "exit": nb.get("exit", [0.0] * k)}
            out.append({"spec": r["track"], "version": ver, "driver": r["driver"], "start": r["start"],
                        "expert": r["expert"], "curve": r["curve"], "points": nb["points"],
                        "apex": nb.get("apex", nb["points"]), "power": nb.get("power", nb["points"]), "notes": notes})
    return out


def python_lap(e: dict, sim: fevo.Simulator, tr: fevo.Track) -> tuple[str, int]:
    """The same lap in Python: notebook2's note application, the same notes and start, stock aero."""
    nb = Notebook(tr.points, e["points"], e["apex"], e["power"], edge_keep=0.2)
    for kind, vals in e["notes"].items():
        nb.notes[kind][:] = vals
    sim.load_program(fevo.from_json((WINNERS / f"{e['driver']}.json").read_text()))
    lap = run_lap(sim, nb, e["start"])
    return lap["status"], lap["steps"]


def build(out: str) -> Path:
    rules = fevo.load_rules(general.RULES)
    setup_d = json.loads((WINNERS / f"{GENERALS[0]}.json").read_text())["setup"]
    setup = [setup_d[k["id"]] for k in rules.setup]
    sim = fevo.Simulator(rules)
    sim.set_setup(setup)
    es = entries()
    tracks, programs, items = {}, {}, []
    for e in es:
        tr = general._track(e["spec"])
        sim.load_track(tr)
        if e["spec"] not in tracks:
            tracks[e["spec"]] = {k: tr.raw[k] for k in ("centerline", "widthLeft", "widthRight", "gates")}
        if e["driver"] not in programs:
            mlp = json.loads((WINNERS / f"{e['driver']}.json").read_text())
            assert mlp["setup"] == setup_d, "every general driver shares the setup"
            programs[e["driver"]] = fevo.from_json(json.dumps(mlp)).packed()
        st, steps = python_lap(e, sim, tr)
        clean = [c for c in e["curve"] if c > 0]
        items.append({**e, "label": f"{e['spec'].replace('gen:', 'g')} · notebook v{e['version']}"
                      + (f" ({fevo.format_lap_time(clean[-1])})" if clean else " (never lapped)"),
                      "python": {"status": st, "steps": steps}})
        print(f"{items[-1]['label']:<40} python lap with notes, start {e['start']}: "
              f"{fevo.format_lap_time(steps) if st == 'finished' else st}", flush=True)
    r = rules.raw
    team = fevo.data.branding()["teams"][0]["real"]
    data = {
        "wasm": base64.b64encode(wasm_bytes()).decode(),
        "rules": {
            "id": rules.id, "agentHz": r["timing"]["agentHz"], "physicsSubsteps": r["timing"]["physicsSubsteps"],
            "maxSteps": r["timing"]["maxSteps"], "stopSpeedMps": float(r["timing"]["stopSpeedMps"]), "stopSteps": r["timing"]["stopSteps"],
            "rayCount": r["sensors"]["rayCount"], "rayLengthM": float(r["sensors"]["rayLengthM"]), "rayFovRad": float(r["sensors"]["rayFovRad"]),
            "speedNormMps": float(r["sensors"]["speedNormMps"]), "initialSpeedMps": float(r["start"]["initialSpeedMps"]),
            "rearSlide": list(rules.rear_slide), "trackLimits": rules.track_limits, "observationSet": rules.observation_set,
            "observationCount": len(rules.observation), "carParamOrder": r["carParamOrder"],
            "car": {"length": r["car"]["lengthM"], "width": r["car"]["widthM"]},
        },
        "car": rules.packed_car(setup),
        "seeds": rules.seeds,
        "tracks": tracks,
        "programs": programs,
        "items": items,
        "colors": team["colors"],
    }
    template = resources.files("fevoracer").joinpath("notebook_lab.html").read_text("utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("<", "\\u003c"))
    path = Path(out)
    path.write_text(html, "utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(prog="notebook-lab", description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--out", default="notebook-lab.html")
    a = ap.parse_args()
    print(f"wrote {build(a.out)}")


if __name__ == "__main__":
    main()
