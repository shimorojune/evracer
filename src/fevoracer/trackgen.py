"""Procedural circuits for training: closed, non-self-intersecting loops with F1-like features.

A random convex hull gives the overall shape; each edge's midpoint is pushed in
or out (hairpins, concave sections), long edges sometimes get a chicane (an S
of two control points), and a closed Catmull-Rom spline through the points,
lightly smoothed and scaled to 3.5-6 km, is the centre line. Widths are 10-15 m
with a gentle variation. A track is kept only if:

* no corner is tighter than `min_radius` and no inside edge folds over itself
  (the track builder's foldedSamples rule: |curvature| x inside width < 1);
* track sections that aren't neighbours along the loop stay at least
  `clearance` metres apart, edge to edge (no crossings, no touching parts).

The file has the simulator's track format (1 m spacing, a gate every 40 m
from the start line, exactly as tools/track-builder bakes), with the start
line at the beginning of the longest straight. Training only: these are not
fevo circuits.

    uv run python -m fevoracer.trackgen --count 8 --out tracks/gen --png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import fevo


def _hull(p: np.ndarray) -> np.ndarray:
    """Convex hull, counter-clockwise (monotone chain)."""
    pts = sorted(map(tuple, p))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    lower: list = []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def _catmull_closed(p: np.ndarray, per: int = 40) -> np.ndarray:
    """Centripetal-ish closed Catmull-Rom (uniform parameter), `per` points per segment."""
    n = len(p)
    out = []
    t = np.linspace(0, 1, per, endpoint=False)[:, None]
    for i in range(n):
        p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    return np.concatenate(out)


def _resample(p: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    seg = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = int(round(total / spacing))
    q = np.linspace(0, total, n, endpoint=False)
    closed = np.vstack([p, p[:1]])
    return np.stack([np.interp(q, s, closed[:, 0]), np.interp(q, s, closed[:, 1])], 1)


def _smooth(p: np.ndarray, passes: int, k: int = 5) -> np.ndarray:
    for _ in range(passes):
        p = sum(np.roll(p, j, 0) for j in range(-k, k + 1)) / (2 * k + 1)
    return p


def _curvature(p: np.ndarray, h: int = 3) -> np.ndarray:
    """Signed curvature (left positive) from heading change over ±h points."""
    d = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    th = np.arctan2(d[:, 1], d[:, 0])
    dth = (np.roll(th, -h) - np.roll(th, h) + np.pi) % (2 * np.pi) - np.pi
    seg = np.linalg.norm(np.roll(p, -h, 0) - np.roll(p, h, 0), axis=1)
    return dth / np.maximum(seg, 1e-6)


def _fillet(poly: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Straights along the polygon's sides, each corner replaced by a circular arc tangent to both (1 m points)."""
    n = len(poly)
    pts = []
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        u1, u2 = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        turn = math.atan2(u1[0] * u2[1] - u1[1] * u2[0], float(np.dot(u1, u2)))
        r = radii[i]
        # The arc may use at most 45% of either side, so neighbouring corners never overlap.
        t = r * math.tan(abs(turn) / 2)
        lim = 0.45 * min(np.linalg.norm(b - a), np.linalg.norm(c - b))
        if t > lim:
            r, t = r * lim / t, lim
        p1 = b - u1 * t
        left = np.array([-u1[1], u1[0]]) * (1 if turn > 0 else -1)
        centre = p1 + left * r
        a0 = math.atan2(p1[1] - centre[1], p1[0] - centre[0])
        steps = max(2, int(abs(turn) * r))
        for k in range(steps + 1):
            ang = a0 + turn * k / steps
            pts.append(centre + r * np.array([math.cos(ang), math.sin(ang)]))
    return _resample(np.array(pts), 1.0)  # the straights are the gaps between consecutive arcs


def _slow_features(poly: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Corners a racing line cannot straighten, as real circuits have: on long sides, a hairpin of 150-172 degrees
    at the tip of a spike (with a near-90-degree corner either side of it) and a sidestep (two tight 90-degree corners
    the other way round, like a first-corner chicane). Returns the new polygon and fixed radii (nan: random)."""
    n = len(poly)
    area = 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    orient = 1.0 if area > 0 else -1.0
    sides = np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1)
    long_ = [int(i) for i in rng.permutation(n) if sides[i] > 350]
    spikes = set(long_[: int(rng.integers(1, 3))])
    steps = set(long_[len(spikes): len(spikes) + int(rng.random() < 0.6)])
    out, fixed = [], []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        out.append(a)
        fixed.append(np.nan)
        e = b - a
        ln = float(np.linalg.norm(e))
        eh = e / ln
        out_n = orient * np.array([eh[1], -eh[0]])  # away from the inside of the loop
        if i in spikes:
            base = a + e * rng.uniform(0.35, 0.65)
            depth = rng.uniform(110, 220)
            turn = math.radians(rng.uniform(150, 172))
            half = max(depth * math.tan((math.pi - turn) / 2), 22.0)
            tip = base + out_n * depth * (1 if rng.random() < 0.6 else -1)
            out += [base - eh * half, tip, base + eh * half]
            fixed += [rng.uniform(15, 35), rng.uniform(14, 19), rng.uniform(15, 35)]
        elif i in steps:
            base = a + e * rng.uniform(0.3, 0.6)
            off = out_n * rng.uniform(30, 42) * (1 if rng.random() < 0.5 else -1)
            out += [base, base + off]
            fixed += [rng.uniform(14, 20), rng.uniform(14, 20)]
    return np.array(out), np.array(fixed)


def generate(seed: int, length=(3500.0, 6000.0), min_radius: float = 14.0, clearance: float = 20.0,
             slow: bool = False) -> dict | None:
    """One circuit, or None when the attempt fails a check (callers retry with another seed).

    slow adds hairpins and sidestep chicanes (_slow_features); without it the generator is unchanged."""
    rng = np.random.default_rng(seed)
    hull = _hull(rng.uniform(-1, 1, (int(rng.integers(8, 16)), 2)))
    if len(hull) < 4:
        return None
    pts = []
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        pts.append(a)
        e = b - a
        ln = np.linalg.norm(e)
        nrm = np.array([e[1], -e[0]]) / max(ln, 1e-9)  # outward for a counter-clockwise hull
        roll = rng.random()
        if roll < 0.2 and ln > 0.5:
            # A chicane: two close points either side of the straight.
            pts.append(a + e * 0.48 + nrm * ln * rng.uniform(0.03, 0.06))
            pts.append(a + e * 0.56 - nrm * ln * rng.uniform(0.03, 0.06))
        elif roll < 0.75:
            pts.append(a + e * rng.uniform(0.35, 0.65) + nrm * ln * rng.uniform(-0.5, 0.15))  # mostly inward
    poly = np.array(pts)
    poly *= rng.uniform(*length) / np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1).sum()
    # Corner radii: about a quarter slow (hairpins, tight corners), half medium, the rest fast sweepers.
    fixed = None
    if slow:
        poly, fixed = _slow_features(poly, np.random.default_rng([seed, 7]))
    radii = np.exp(rng.uniform(np.log(15), np.log(320), len(poly)))
    if fixed is not None:
        radii = np.where(np.isnan(fixed), radii, fixed)
    p = _fillet(poly, radii)
    p = _resample(_smooth(p, passes=1, k=2), 1.0)
    if rng.random() < 0.5:
        p = p[::-1].copy()
    n = len(p)
    k = _curvature(p)
    if np.max(np.abs(k)) > 1.0 / min_radius:
        return None
    half = rng.uniform(5.0, 7.5)
    s = np.arange(n) / n * 2 * np.pi
    var = sum(rng.uniform(-0.35, 0.35) * np.sin(f * s + rng.uniform(0, 2 * np.pi)) for f in (2, 3, 5))
    wl = np.clip(half + var + rng.uniform(-0.3, 0.3, n) * 0, 4.5, 8.0)
    wr = np.clip(half + var, 4.5, 8.0)
    inside = np.where(k > 0, wl, wr)
    if np.any(np.abs(_curvature(p, 3)) * inside >= 1):
        return None
    # Clearance: points far apart along the loop must stay apart in space, edge to edge.
    step = 4
    q = p[::step]
    m = len(q)
    d = np.sqrt(((q[:, None, :] - q[None, :, :]) ** 2).sum(-1))
    idx = np.arange(m)
    along = np.abs(idx[:, None] - idx[None, :]) * step
    along = np.minimum(along, n - along)
    wmax = np.maximum(wl, wr)[::step]
    need = wmax[:, None] + wmax[None, :] + clearance
    far = along > 3 * (need + 40)
    if np.any(far & (d < need)):
        return None
    # Start at the beginning of the longest straight: the 300 m window with the least curvature.
    win = 300
    ka = np.abs(k)
    roll = np.array([ka.take(range(i, i + win), mode="wrap").mean() for i in range(0, n, 10)])
    start = int(np.argmin(roll)) * 10 + 30
    p = np.roll(p, -start, 0)
    wl = np.roll(wl, -start)
    wr = np.roll(wr, -start)
    seg = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1)
    length_m = float(seg.sum())
    gate_count = max(1, round(length_m / 40.0))
    gates = [round(i * n / gate_count) for i in range(gate_count)]
    r = lambda v: [round(float(x), 3) for x in v]  # noqa: E731
    body = {
        "schema": 1, "circuitId": f"{'slow' if slow else 'gen'}-{seed}", "revision": 1,
        "source": {"kind": "generated", "seed": seed, "generator": "fevoracer.trackgen", **({"slow": True} if slow else {})},
        "officialLengthM": round(length_m), "lengthM": round(length_m, 3), "spacingM": round(length_m / n, 3),
        "centerline": r(p.ravel()), "widthLeft": r(wl), "widthRight": r(wr), "gates": gates,
        "bounds": {"minX": round(float((p[:, 0] - 8).min()), 3), "minY": round(float((p[:, 1] - 8).min()), 3),
                   "maxX": round(float((p[:, 0] + 8).max()), 3), "maxY": round(float((p[:, 1] + 8).max()), 3)},
    }
    body["sha256"] = hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()
    return body


def make(count: int, first_seed: int = 1, **kw) -> list[dict]:
    out, seed = [], first_seed
    while len(out) < count:
        t = generate(seed, **kw)
        if t is not None:
            out.append(t)
        seed += 1
    return out


def features(t: dict) -> dict:
    """Corner counts by radius, for a quick look at what a track asks of the driver."""
    p = np.array(t["centerline"]).reshape(-1, 2)
    k = np.abs(_curvature(p, 6))
    radius = 1 / np.maximum(k, 1e-9)
    corners = {"hairpin <30 m": 0, "slow 30-80 m": 0, "medium 80-200 m": 0, "fast 200-600 m": 0}
    edges = [(0, 30), (30, 80), (80, 200), (200, 600)]
    # Count separate corner segments (runs of points under each radius band's upper limit).
    for (lo, hi), name in zip(edges, corners):
        inside = (radius < hi) & (radius >= lo)
        corners[name] = int(np.sum(inside & ~np.roll(inside, 1)))
    straight = (radius > 1000).astype(int)
    runs, cur = [], 0
    for v in np.concatenate([straight, straight]):
        cur = cur + 1 if v else 0
        runs.append(cur)
    return {"lengthM": t["lengthM"], "widthM": round(float(np.mean(np.array(t["widthLeft"]) + np.array(t["widthRight"]))), 1),
            "longestStraightM": int(min(max(runs), len(straight))), **corners}


def main() -> None:
    ap = argparse.ArgumentParser(prog="trackgen", description="Procedural training circuits in the simulator's track format")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="tracks/gen")
    ap.add_argument("--png", action="store_true", help="also draw them into one overview image (SVG)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tracks = make(a.count, a.seed)
    for t in tracks:
        (out / f"{t['circuitId']}.track.json").write_text(json.dumps(t) + "\n")
        print(t["circuitId"], features(t))
    if a.png:
        draw(tracks, out / "overview.svg")
        print(f"wrote {out / 'overview.svg'}")


def draw(tracks: list[dict], path: Path) -> None:
    """An overview of the tracks as one SVG (no plotting dependency): surface, edges and the start line."""
    cols, cell = 4, 300
    rows = math.ceil(len(tracks) / cols)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{cols * cell}" height="{rows * (cell + 24)}" '
             f'style="background:#f5f4ef;font:12px system-ui">']
    for i, t in enumerate(tracks):
        p = np.array(t["centerline"]).reshape(-1, 2)
        d = np.roll(p, -1, 0) - np.roll(p, 1, 0)
        d /= np.linalg.norm(d, axis=1)[:, None]
        nl = np.stack([-d[:, 1], d[:, 0]], 1)
        left = p + nl * np.array(t["widthLeft"])[:, None]
        right = p - nl * np.array(t["widthRight"])[:, None]
        allp = np.vstack([left, right])
        lo, hi = allp.min(0), allp.max(0)
        sc = (cell - 30) / max(hi - lo)
        ox, oy = (i % cols) * cell + 15, (i // cols) * (cell + 24) + 30
        tf = lambda q: [(ox + (x - lo[0]) * sc, oy + (hi[1] - y) * sc) for x, y in q[::3]]  # noqa: E731
        path_ = lambda q: "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in tf(q)) + " Z"  # noqa: E731
        parts.append(f'<path d="{path_(left)} {path_(right)}" fill="#3b3b3e" fill-rule="evenodd" stroke="#999" stroke-width="0.6"/>')
        g0 = t["gates"][0]
        (x0, y0), (x1, y1) = tf(left[g0:g0 + 1])[0], tf(right[g0:g0 + 1])[0]
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#e5484d" stroke-width="3"/>')
        f = features(t)
        parts.append(f'<text x="{ox}" y="{oy - 10}">{t["circuitId"]}: {f["lengthM"] / 1000:.2f} km, {f["widthM"]} m wide, '
                     f'{f["hairpin <30 m"]} hairpins</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def track(t: dict) -> fevo.Track:
    """A generated track as the SDK's Track (Simulator.load_track takes it directly)."""
    return fevo.data.Track(t)


if __name__ == "__main__":
    main()
