"""How far is a driver from the limit, and where does the time go?

Compares one official lap against two quasi-steady-state laps on the
reference racing line (fevo.reference, from tools/calibrate):

* point-mass: the 1:07.1-style reference itself (flying lap, both axles at
  peak grip together, all four tyres braking, default setup), ported from
  tools/calibrate/src/line.ts and checked against its lap time;
* vehicle: the same line with the simulated car's own limits and the
  driver's setup: a standing start, per-axle cornering grip (front and rear
  each carry their share of the cornering force, as the bicycle model does),
  fixed brake bias with longitudinal load transfer, rear-wheel traction on the
  rear friction circle. Still optimistic (no yaw transients, steering rate or
  driver reaction), so a lap the simulator can't quite reach.

Time is attributed per centre-line point (the agent's time to get from point
i to i + 1 minus the vehicle lap's), so sector and category losses add up to
the whole gap exactly. Categories follow what limits the vehicle lap at that
point: braking, cornering (lateral grip), traction (rear grip on exit) or
power (straights).

    uv run python -m fevoracer.gap winners/solo2safe-spielberg.json --rules solo-2 --seed 1
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np

import fevo

KMH = 3.6


# --------------------------------------------------------------------------- reference line


def ref_line(track: fevo.Track) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Racing line points, the centre line and its left normals, and the line's offsets (left positive)."""
    c = np.array(track.raw["centerline"]).reshape(-1, 2)
    t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    nl = np.stack([-t[:, 1], t[:, 0]], 1)
    off = np.array(fevo.reference(track.id)["offset"])
    return c + nl * off[:, None], c, nl, off, t


def curvature(p: np.ndarray, h: int = 3) -> np.ndarray:
    """Circle through each point and its neighbours h away (as tools/calibrate)."""
    a, b, c = np.roll(p, h, 0), p, np.roll(p, -h, 0)
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    d = np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(c - a, axis=1)
    return np.where(d > 0, 2 * area2 / np.where(d > 0, d, 1), 0.0)


# --------------------------------------------------------------------------- quasi-steady-state laps


@dataclass
class Lap:
    v: np.ndarray  # speed at each line point, n + 1 values (the last is the finish line)
    vmax: np.ndarray  # cornering limit per point
    phase: list[str]  # what limits the lap at each point
    seg_t: np.ndarray  # time from point i to i + 1
    lap_s: float


def qss(p: np.ndarray, car: dict[str, float], vehicle: bool, standing: bool, h_curv: int = 3, corner_scale: float = 1.0) -> Lap:
    """corner_scale < 1 keeps a margin on cornering speeds only; braking and acceleration stay at the limit."""
    n = len(p)
    k = curvature(p, h_curv)
    ds = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1)
    m, g, rho = car["massKg"], car["gravityMps2"], car["airDensityKgM3"]
    L, a = car["wheelbaseM"], car["cgToFrontM"]
    b, h = L - a, car["cgHeightM"]
    muF, muR, bal = car["tyreMuFront"], car["tyreMuRear"], car["aeroBalanceFront"]
    P, Bmax, bias = car["powerW"], car["maxBrakeForceN"], car["brakeBiasFront"]

    def loads(v: float, ax: float = 0.0) -> tuple[float, float]:
        df = 0.5 * rho * car["downforceAreaM2"] * v * v
        tr = m * ax * h / L if vehicle else 0.0
        return max(0.0, m * g * b / L + df * bal - tr), max(0.0, m * g * a / L + df * (1 - bal) + tr)

    def drag(v: float) -> float:
        return 0.5 * rho * car["dragAreaM2"] * v * v + car["rollingResistance"] * m * g

    def spare(need: float, cap: float) -> float:
        return math.sqrt(max(0.0, 1 - (need / cap) ** 2)) if cap > 0 else 0.0

    def vlim(ki: float) -> float:
        v = 90.0
        for _ in range(40):
            f, r = loads(v)
            if vehicle:  # each axle carries its share of the cornering force (b/L front, a/L rear)
                cap = min(muF * f / (b / L), muR * r / (a / L))
            else:
                cap = muF * f + muR * r
            v = min(110.0, math.sqrt(cap / m / max(ki, 1e-6)))
        return v

    def accel(v: float, ki: float) -> tuple[float, bool]:
        """Forward acceleration and whether traction (not power) limits it."""
        ax = 0.0
        for _ in range(4 if vehicle else 1):
            f, r = loads(v, ax)
            fy = m * v * v * ki
            need_r = fy * a / L if vehicle else fy
            cap_r = muR * r
            cap_all = cap_r if vehicle else muF * f + muR * r
            trac = muR * r * spare(need_r if vehicle else fy, cap_all if not vehicle else cap_r)
            if not vehicle:
                trac = muR * r * spare(fy, muF * f + muR * r)
            drive = min(P / max(v, 1.0), trac)
            ax = (drive - drag(v)) / m
        return ax, trac < P / max(v, 1.0)

    def decel(v: float, ki: float) -> float:
        ax = 0.0
        for _ in range(4 if vehicle else 1):
            f, r = loads(v, -ax)
            fy = m * v * v * ki
            if vehicle:
                brake = min(Bmax, muF * f * spare(fy * b / L, muF * f) / bias, muR * r * spare(fy * a / L, muR * r) / (1 - bias))
            else:
                brake = min(Bmax, muF * f + muR * r) * spare(fy, muF * f + muR * r)
            ax = (brake + drag(v)) / m
        return ax

    vmax = np.array([vlim(ki) for ki in k] + [vlim(k[0])]) * corner_scale
    v = vmax.copy()
    if standing:
        v[0] = 0.0
    trac_lim = [False] * (n + 1)
    for _ in range(1 if standing else 3):
        for i in range(n):
            acc, tl = accel(v[i], k[i])
            nv = math.sqrt(max(0.0, v[i] ** 2 + 2 * acc * ds[i]))
            if nv < v[i + 1]:
                v[i + 1], trac_lim[i] = nv, tl
            if not standing and i == n - 1:
                v[0] = min(v[0], v[n])
        if not standing:
            v[n] = v[0]
        for i in range(n - 1, -1, -1):
            nv = math.sqrt(v[i + 1] ** 2 + 2 * decel(v[i + 1], k[(i + 1) % n]) * ds[i])
            if nv < v[i] and not (standing and i == 0):
                v[i] = nv
        if not standing:
            v[n] = v[0]
    seg_t = ds / np.maximum(0.5, (v[:-1] + v[1:]) / 2)
    phase = []
    for i in range(n):
        if v[i + 1] < v[i] - 1e-6:
            phase.append("braking")
        elif v[i] >= vmax[i] * 0.97:
            phase.append("cornering")
        else:
            phase.append("traction" if trac_lim[i] else "power")
    return Lap(v, vmax, phase, seg_t, float(seg_t.sum()))


# --------------------------------------------------------------------------- the agent's lap


def drive(sim: fevo.Simulator, seed: int, c: np.ndarray, nl: np.ndarray, hz: int, policy=None) -> dict[str, np.ndarray]:
    """One lap from an official start, by the loaded program or by policy(sim) -> (steer, throttle)."""
    call = sim.core.call
    obs = sim.reset(seed)
    rows = []
    status = "running"
    while status == "running":
        if policy is None:
            status = sim.agent_step()
            thr = call("getAgentThrottle")
        else:
            steer, thr = policy(sim, obs)
            status = sim.step(steer, thr)
            obs = sim.observation()
        i = call("getProgressIndex")
        x, y = call("getX"), call("getY")
        rows.append((i, call("getVx"), max(-1.0, min(1.0, thr)), call("getFrontSlide"), call("getRearSlide"),
                     float(np.dot([x, y] - c[i % len(c)], nl[i % len(c)]))))
    r = np.array(rows)
    return {"status": status, "steps": len(rows), "idx": r[:, 0].astype(int), "v": r[:, 1], "thr": r[:, 2], "fs": r[:, 3],
            "rs": r[:, 4], "off": r[:, 5], "hz": hz}


def time_at(lap: dict[str, np.ndarray], n: int) -> np.ndarray:
    """Seconds from the start to the first step at or past each centre-line point (n + 1 values)."""
    t = np.full(n + 1, np.nan)
    best = -1
    for s, i in enumerate(lap["idx"]):
        # progress index runs 0..n-1; count laps so the finish reads as n
        if i > best and i - best < n // 2:
            t[best + 1 : i + 1] = (s + 1) / lap["hz"]
            best = i
    t[0] = 0.0
    t[n] = lap["steps"] / lap["hz"] if lap["status"] == "finished" else np.nan
    return t


# --------------------------------------------------------------------------- report


def corners(v: np.ndarray, drop_kmh: float = 15.0, merge_m: int = 120) -> list[tuple[int, int]]:
    """(braking point, apex) per corner: speed minima at least drop_kmh below the preceding maximum."""
    n = len(v) - 1
    out: list[tuple[int, int]] = []
    hi_i, i = 0, 1
    while i < n:
        if v[i] > v[hi_i]:
            hi_i = i
        if v[i] <= v[i - 1] and v[i] <= v[i + 1] and (v[hi_i] - v[i]) * KMH >= drop_kmh:
            j = i
            while j + 1 < n and v[j + 1] <= v[j] + 1e-9:
                j += 1
            apex = int(np.argmin(v[i : j + 1])) + i
            if out and apex - out[-1][1] < merge_m:
                if v[apex] < v[out[-1][1]]:
                    out[-1] = (out[-1][0], apex)
            else:
                out.append((hi_i, apex))
            hi_i, i = j, j + 1
            continue
        i += 1
    return out


def report(name: str, track: fevo.Track, rules: fevo.Rules, setup: list[float] | None, lap: dict, seed: int,
           json_path: str | None = None) -> dict:
    """Print the waterfall, the category split and the sector table for one recorded lap (see drive())."""
    hz = rules.agent_hz
    car_default, car_agent = rules.car, rules.car_for(setup)
    p, c, nl, roff, _ = ref_line(track)
    n = len(p)
    # The 1:07.1 reference uses curvature over ±3 m; the shipped offsets are rounded to the millimetre,
    # which makes that noisy at fast apexes, so everything else uses ±10 m (exact on a true circle).
    pm_fly3 = qss(p, car_default, vehicle=False, standing=False)
    pm_fly = qss(p, car_default, vehicle=False, standing=False, h_curv=10)
    pm_std = qss(p, car_default, vehicle=False, standing=True, h_curv=10)
    vh_fly_def = qss(p, car_default, vehicle=True, standing=False, h_curv=10)
    vh_fly = qss(p, car_agent, vehicle=True, standing=False, h_curv=10)
    vh = qss(p, car_agent, vehicle=True, standing=True, h_curv=10)
    ta = time_at(lap, n)
    if lap["status"] != "finished" or np.isnan(ta).any():
        raise SystemExit(f"seed {seed} {lap['status']}; pick a seed that finishes")
    agent_s = lap["steps"] / hz
    tv = np.concatenate([[0.0], np.cumsum(vh.seg_t)])
    tp = np.concatenate([[0.0], np.cumsum(pm_fly3.seg_t)])  # the published 1:07.1 lap, for the "vs 1:07" column
    loss = np.diff(ta) - vh.seg_t  # per point, sums to agent - vehicle lap

    print(f"{name} at {track.name} ({rules.id}), seed {seed}: {fevo.format_lap_time(lap['steps'], hz)}")
    print(f"setup {dict(zip([k['id'] for k in rules.setup], setup)) if setup else 'default'}: front mu {car_agent['tyreMuFront']:.3f}, rear mu {car_agent['tyreMuRear']:.3f}, "
          f"downforce {car_agent['downforceAreaM2']:.2f} m2, drag {car_agent['dragAreaM2']:.3f} m2, brake bias {car_agent['brakeBiasFront']:.3f}")
    print("\nFrom the 1:07.1 reference to this lap")
    steps = [
        ("point-mass reference as published (curvature over ±3 m)", pm_fly3.lap_s),
        ("point-mass reference, flying lap (default setup)", pm_fly.lap_s),
        ("  + standing start", pm_std.lap_s),
        ("vehicle limits, flying lap, default setup", vh_fly_def.lap_s),
        ("vehicle limits, flying lap, this setup", vh_fly.lap_s),
        ("vehicle limits, standing start, this setup  (achievable estimate)", vh.lap_s),
        ("this lap", agent_s),
    ]
    prev = None
    for name, s in steps:
        d = "" if prev is None else f"  ({s - prev:+.2f} s)"
        print(f"  {name:<68} {fevo.format_lap_time(round(s * hz), hz)}{d}")
        prev = s
    print(f"  standing start alone costs the point mass {pm_std.lap_s - pm_fly.lap_s:.2f} s; "
          f"vehicle limits cost it {vh_fly_def.lap_s - pm_fly.lap_s:.2f} s on a flying lap")

    cats = ["braking", "cornering", "traction", "power"]
    by_cat = {cname: float(sum(loss[i] for i in range(n) if vh.phase[i] == cname)) for cname in cats}
    dist = {cname: sum(1 for i in range(n) if vh.phase[i] == cname) for cname in cats}
    print(f"\nWhere the {agent_s - vh.lap_s:.2f} s to the achievable lap goes (by what limits the vehicle lap there)")
    # What the driver does there: speed deficit and pedal, averaged over the driver's steps in each phase.
    phase_of_step = [vh.phase[i % n] for i in lap["idx"]]
    for cname in cats:
        sel = np.array([ph == cname for ph in phase_of_step])
        dv = (vh.v[lap["idx"][sel] % n] - lap["v"][sel]) * KMH if sel.any() else np.array([0.0])
        thr = lap["thr"][sel] if sel.any() else np.array([0.0])
        print(f"  {cname:<10} {by_cat[cname]:+6.2f} s   over {dist[cname]:5d} m of track   driver {dv.mean():5.1f} km/h slower, "
              f"throttle mean {thr.mean():+.2f}, full {100 * (thr >= 0.98).mean():3.0f}%, braking {100 * (thr < -0.05).mean():3.0f}%")

    cs = corners(vh.v)
    bounds = [0] + [bp for bp, _ in cs] + [n]
    print(f"\nSectors: braking point to braking point ({len(cs)} corners). Times in s, speeds in km/h, distances in m along the centre line.")
    print("  early: metres the driver starts its last braking run before the achievable braking point (negative = later);"
          " v@bp: speed arriving at the achievable braking point; pm: point-mass apex speed")
    hdr = (f"{'sector':<15}{'lap':>7}{'achv':>7}{'loss':>7}{'vs 1:07':>8} | {'min v':>6}{'achv':>6}{'pm':>6} | "
           f"{'brake@':>7}{'achv@':>7}{'early':>6}{'v@bp':>6}{'achv':>6} | {'full thr':>8}{'achv acc':>9} | {'line dev':>8} | {'spin s':>6}{'lock s':>7}")
    print(hdr)
    rows = []
    for s in range(len(bounds) - 1):
        i0, i1 = bounds[s], bounds[s + 1]
        steps_in = (lap["idx"] >= i0) & (lap["idx"] < i1)
        name = "start" if s == 0 else f"T{s} @ {i0} m"
        apex = cs[s - 1][1] if s > 0 else None
        a_t, v_t, r_t = ta[i1] - ta[i0], tv[i1] - tv[i0], tp[i1] - tp[i0]
        row = {"sector": name, "from_m": i0, "to_m": i1, "agent_s": a_t, "achievable_s": v_t, "loss_s": a_t - v_t, "vs_ref_s": a_t - r_t}
        if apex is not None:
            win = steps_in & (np.abs(lap["idx"] - apex) < 80)
            row["min_kmh"] = float(lap["v"][win].min() * KMH) if win.any() else float("nan")
            row["achv_min_kmh"] = float(vh.v[apex] * KMH)
            row["ref_min_kmh"] = float(pm_fly.v[apex] * KMH)
            # The driver's braking point: the start of its last braking run before the apex (gaps up to 0.25 s allowed).
            prev_apex = cs[s - 2][1] if s > 1 else 0
            br = np.where((lap["idx"] > prev_apex) & (lap["thr"] < -0.05) & (lap["idx"] <= apex))[0]
            if len(br):
                j = len(br) - 1
                while j > 0 and br[j] - br[j - 1] <= hz // 4:
                    j -= 1
                row["brake_at_m"] = int(lap["idx"][br[j]])
            else:
                row["brake_at_m"] = None
            row["achv_brake_at_m"] = i0
            at_bp = np.where(lap["idx"] >= i0)[0]
            row["v_at_bp_kmh"] = float(lap["v"][at_bp[0]] * KMH) if len(at_bp) else float("nan")
            row["achv_v_at_bp_kmh"] = float(vh.v[i0] * KMH)
            row["brakes_early_m"] = (i0 - row["brake_at_m"]) if row["brake_at_m"] is not None else None
        full = steps_in & (lap["thr"] >= 0.98)
        row["full_throttle_m"] = int(len(np.unique(lap["idx"][full])))
        row["achv_accel_m"] = sum(1 for i in range(i0, i1) if vh.phase[i] in ("traction", "power"))
        row["line_dev_m"] = float(np.mean(np.abs(lap["off"][steps_in] - roff[lap["idx"][steps_in] % n]))) if steps_in.any() else 0.0
        row["spin_s"] = float(((lap["rs"] > 0) & (lap["thr"] >= 0) & steps_in).sum() / hz)
        row["lock_s"] = float(((lap["thr"] < 0) & ((lap["rs"] > 0) | (lap["fs"] > 0)) & steps_in).sum() / hz)
        rows.append(row)
        f = lambda x, w=6, d=0: (f"{x:>{w}.{d}f}" if isinstance(x, (int, float)) and x is not None and not (isinstance(x, float) and math.isnan(x)) else f"{'-':>{w}}")  # noqa: E731
        print(f"{name:<15}{a_t:>7.2f}{v_t:>7.2f}{a_t - v_t:>+7.2f}{a_t - r_t:>+8.2f} | {f(row.get('min_kmh'))}{f(row.get('achv_min_kmh'))}{f(row.get('ref_min_kmh'))} | "
              f"{f(row.get('brake_at_m'), 7)}{f(row.get('achv_brake_at_m'), 7)}{f(row.get('brakes_early_m'), 6)}"
              f"{f(row.get('v_at_bp_kmh'))}{f(row.get('achv_v_at_bp_kmh'))} | "
              f"{row['full_throttle_m']:>8}{row['achv_accel_m']:>9} | {row['line_dev_m']:>8.2f} | {row['spin_s']:>6.2f}{row['lock_s']:>7.2f}")
    worst = sorted(rows, key=lambda r: -r["loss_s"])[:3]
    print("\nBiggest sector losses to the achievable lap: " + ", ".join(f"{r['sector']} {r['loss_s']:+.2f} s" for r in worst))
    if json_path:
        json.dump({"waterfall": dict(steps), "by_category": by_cat, "sectors": rows}, open(json_path, "w"), indent=1, default=float)

    return {"waterfall": dict(steps), "by_category": by_cat, "sectors": rows}


def main() -> None:
    ap = argparse.ArgumentParser(prog="gap", description="Where an agent loses time against the reference and the vehicle limit")
    ap.add_argument("file")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("--rules", default="solo-1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", help="also write the numbers here")
    a = ap.parse_args()

    mlp = json.load(open(a.file))
    sim = fevo.Simulator(a.rules)
    track = sim.load_track(a.track)
    prog = fevo.from_json(json.dumps(mlp))
    sim.load_program(prog)
    rules = sim.rules
    setup = [mlp["setup"][k["id"]] for k in rules.setup] if "setup" in mlp else None
    _, c, nl, _, _ = ref_line(track)
    lap = drive(sim, a.seed, c, nl, rules.agent_hz)
    report(a.file, track, rules, setup, lap, a.seed, a.json)


if __name__ == "__main__":
    main()
