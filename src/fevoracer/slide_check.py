"""How much grip does a sliding tyre keep in the simulator? (Rohit: a sliding car should not stop at full grip.)

1. The force curves past peak, from the rules' car parameters and sim-core's vehicle.ts: lateral force against
   slip angle (Pacejka, per axle), the locked or spinning cases, and the share of peak kept at 2x, 4x and 8x the
   peak slip angle.
2. Spielberg T1 with the notes as the page listed them in Rohit's recording (start 2): per step, each axle's slip
   angle against its peak, the lateral force as a share of peak, the pedal and the deceleration.

    uv run python -m fevoracer.slide_check
"""

from __future__ import annotations

import json
import math

import numpy as np

import fevo

from . import general
from .notebook2 import Notebook
from .paths import WINNERS

SLIDE_MU, SLIDE_LATERAL = 0.85, 0.25  # sim-core vehicle.ts: a locked or (solo-1) spinning tyre
REAR_SLIDE_MU = 0.9  # solo-2 on: a rear spinning under power, combined slip


def lateral(alpha: float, b: float, c: float, e: float) -> float:
    ba = b * alpha
    return math.sin(c * math.atan(ba - e * (ba - math.atan(ba))))


def peak(b: float, c: float, e: float) -> tuple[float, float]:
    xs = np.radians(np.arange(0.01, 90, 0.01))
    fs = np.array([lateral(x, b, c, e) for x in xs])
    i = int(np.argmax(fs))
    return float(xs[i]), float(fs[i])


def main() -> None:
    setup_d = json.loads((WINNERS / "general-solo4-es.json").read_text())["setup"]
    for rid in ("solo-3", "solo-4"):
        rules = fevo.load_rules(rid)
        car = rules.car_for([setup_d[k["id"]] for k in rules.setup])
        print(f"\n{rid}: rear tyre model {rules.rear_slide}, Pacejka C {car['pacejkaC']}, E {car['pacejkaE']}")
        for axle, key in (("front", "pacejkaBFront"), ("rear", "pacejkaBRear")):
            b = car[key]
            ap, fp = peak(b, car["pacejkaC"], car["pacejkaE"])
            shares = [lateral(k * ap, b, car["pacejkaC"], car["pacejkaE"]) / fp for k in (2, 4, 8)]
            print(f"  {axle} (B {b}): peak at {math.degrees(ap):.1f} deg; past it, {shares[0]:.0%} at 2x ({math.degrees(2 * ap):.0f} deg), "
                  f"{shares[1]:.0%} at 4x ({math.degrees(4 * ap):.0f} deg), {shares[2]:.0%} at 8x ({math.degrees(8 * ap):.0f} deg), "
                  f"{lateral(math.pi / 2, b, car['pacejkaC'], car['pacejkaE']) / fp:.0%} fully sideways")
        print(f"  locked wheel (front or rear, braking past the limit): {SLIDE_MU:.0%} of peak along the wheel, "
              f"cornering force capped at {SLIDE_LATERAL * SLIDE_MU:.0%} of peak")
        if rules.rear_slide[0]:
            print(f"  rear spinning under power (combined slip): {REAR_SLIDE_MU:.0%} of peak, shared between drive and side force")
        print("  within the limit, braking/drive and cornering share each axle's grip on a friction circle; downforce "
              "comes from the forward speed only, so a car going sideways also loses its aero grip")

    # 2. The clip.
    rules = fevo.load_rules("solo-4")
    setup = [setup_d[k["id"]] for k in rules.setup]
    car = rules.car_for(setup)
    sim = fevo.Simulator(rules)
    sim.set_setup(setup)
    sim.load_track("spielberg")
    sim.load_program(fevo.from_json((WINNERS / "general-solo4-mixd-r30.json").read_text()))
    nbj = json.loads((WINNERS / "notebook2" / "spielberg.json").read_text())["notebook"]
    nb = Notebook(sim.track.points, nbj["points"], nbj["apex"], nbj["power"], edge_keep=0.2)
    for km, (kind, val) in {0.35: ("brake", -11.0), 1.24: ("brake", -18.0), 2.09: ("brake", -15.0), 2.62: ("line", -0.02)}.items():
        nb.notes[kind][int(np.argmin([abs(p / 1000 - km) for p in nb.points]))] = val
    c = sim.core.call
    L, a = car["wheelbaseM"], car["cgToFrontM"]
    bb = L - a
    apF, fpF = peak(car["pacejkaBFront"], car["pacejkaC"], car["pacejkaE"])
    apR, fpR = peak(car["pacejkaBRear"], car["pacejkaC"], car["pacejkaE"])
    n = sim.track.points
    obs, st, rows = sim.reset(2), "running", []
    while st == "running" and sim.progress_index < 560:
        j = sim.progress_index % n
        steer, own = sim.evaluate_program(obs)
        own = float(np.clip(own, -1, 1))
        steer, thr = nb.apply(j, float(steer), own, obs)
        vx, vy, r, delta = c("getVx"), c("getVy"), c("getYawRate"), c("getSteerAngle")
        st = sim.step(float(np.clip(steer, -1, 1)), float(np.clip(thr, -1, 1)))
        obs = sim.observation()
        vxs = max(vx, 1.0)
        aF = delta - math.atan2(vy + a * r, vxs)
        aR = -math.atan2(vy - bb * r, vxs)
        rows.append((j, math.hypot(vx, vy) * 3.6, float(np.clip(thr, -1, 1)), float(np.clip(steer, -1, 1)), aF, aR,
                     c("getFrontSlide"), c("getRearSlide"), (math.hypot(vx, vy) - math.hypot(c("getVx"), c("getVy"))) * 60 / 9.81))
    print("\nSpielberg T1, notes as in the recording, start 2: each axle's slip angle (x its peak) and lateral force (share of peak)")
    print("  point  km/h  pedal  steer   front slip / force    rear slip / force    locked F/R   decel")
    for row in rows[::6]:
        j, v, thr, steer, aF, aR, fs, rs, dec = row
        if j < 300:
            continue
        sf = lateral(abs(aF), car["pacejkaBFront"], car["pacejkaC"], car["pacejkaE"]) / fpF
        sr = lateral(abs(aR), car["pacejkaBRear"], car["pacejkaC"], car["pacejkaE"]) / fpR
        print(f"  {j:5d} {v:5.0f}  {thr:+.2f}  {steer:+.2f}   {abs(aF) / apF:4.1f}x  {sf:4.0%}          {abs(aR) / apR:4.1f}x  {sr:4.0%}"
              f"          {fs:.2f}/{rs:.2f}   {dec:5.2f} g")
    past = [r for r in rows if r[0] >= 300 and (abs(r[4]) > apF or abs(r[5]) > apR)]
    print(f"  steps from point 300 with an axle past its peak slip: {len(past)} of {sum(1 for r in rows if r[0] >= 300)}"
          + (f"; furthest past: front {max(abs(r[4]) for r in past) / apF:.1f}x, rear {max(abs(r[5]) for r in past) / apR:.1f}x" if past else ""))


if __name__ == "__main__":
    main()
