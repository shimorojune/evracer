"""Build the aero lab: a live, in-browser lap of one agent where the car's aero can be changed mid-lap.

The page embeds the fevo SDK's exact sim.wasm, the track, the rules and the
packed agent, and drives the simulator through its scratch buffers as the
Python SDK does, so a lap at stock aero is bit-for-bit the official lap.
Changing a slider rewrites the car parameters with setCarParams, which the
simulator reads every physics substep: no reset, no change to the simulator.

    uv run python -m fevoracer.aero_live winners/dagger-solo2-tuned-spielberg.json --rules solo-2 -o aero-lab.html
"""

from __future__ import annotations

import argparse
import base64
import json
from importlib import resources
from pathlib import Path

import fevo
from fevo._wasm import wasm_bytes


def build(agent_file: str, out: str, track_id: str, rules_id: str) -> Path:
    mlp = json.loads(Path(agent_file).read_text())
    rules = fevo.load_rules(rules_id)
    track = fevo.load_track(track_id)
    program = fevo.from_json(json.dumps(mlp))
    setup = [mlp["setup"][k["id"]] for k in rules.setup] if "setup" in mlp else None
    official = fevo.evaluate(program, track_id, rules_id)
    r = rules.raw
    team = fevo.data.branding()["teams"][0]["real"]
    data = {
        "agent": Path(agent_file).name,
        "trackName": track.name,
        "wasm": base64.b64encode(wasm_bytes()).decode(),
        "rules": {
            "id": rules.id, "agentHz": r["timing"]["agentHz"], "physicsSubsteps": r["timing"]["physicsSubsteps"],
            "maxSteps": r["timing"]["maxSteps"], "stopSpeedMps": float(r["timing"]["stopSpeedMps"]), "stopSteps": r["timing"]["stopSteps"],
            "rayCount": r["sensors"]["rayCount"], "rayLengthM": float(r["sensors"]["rayLengthM"]), "rayFovRad": float(r["sensors"]["rayFovRad"]),
            "speedNormMps": float(r["sensors"]["speedNormMps"]), "initialSpeedMps": float(r["start"]["initialSpeedMps"]),
            "rearSlide": list(rules.rear_slide), "carParamOrder": r["carParamOrder"],
            "car": {"length": r["car"]["lengthM"], "width": r["car"]["widthM"]},
        },
        "car": rules.packed_car(setup),
        "seeds": rules.seeds,
        "track": {k: track.raw[k] for k in ("centerline", "widthLeft", "widthRight", "gates")},
        "program": program.packed(),
        "official": [{"seed": l.seed, "status": l.status, "steps": l.steps, "finished": l.finished} for l in official.laps],
        "colors": team["colors"],
    }
    template = resources.files("fevoracer").joinpath("aero_live.html").read_text("utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("<", "\\u003c"))
    path = Path(out)
    path.write_text(html, "utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(prog="aero-live", description="Live in-browser lap with aero sliders that apply mid-lap")
    ap.add_argument("file")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("--rules", default="solo-2")
    ap.add_argument("-o", "--out", default="aero-lab.html")
    a = ap.parse_args()
    print(f"wrote {build(a.file, a.out, a.track, a.rules)}")


if __name__ == "__main__":
    main()
