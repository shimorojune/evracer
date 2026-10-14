"""Real circuits from the TUM racetrack-database, converted for training (fevo-racer1 only).

Source: github.com/TUMFTM/racetrack-database (LGPL-3.0), raw CSVs of centre
line x, y and right/left track widths. The conversion mirrors
tools/track-builder's bake: a closed Catmull-Rom spline through the points,
resampled to 1 m with the widths interpolated, a gate every 40 m from the
CSV's first point, and the folded-edge check. Each track file records the
source URL and the licence. These are training and test tracks, not fevo
circuits.

Held out, never trained on: Monza and Austin (shipped with fevo), plus Montreal
(a street-style circuit) and Spa (high speed).

    uv run python -m fevoracer.realtracks
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import fevo

from .paths import ROOT
from .trackgen import _curvature

RAW = ROOT / "data" / "tumftm" / "raw"
OUT = ROOT / "tracks" / "real"
URL = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master/tracks/{}.csv"
HOLDOUT = {"monza", "austin", "montreal", "spa"}
#: Left out of training: Shanghai's tight hairpin folds the inside edge over itself in this data (3 samples).
EXCLUDED = {"shanghai"}


def parse(text: str) -> np.ndarray:
    rows = [ln.split(",") for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    return np.array([[float(v) for v in r[:4]] for r in rows])  # x, y, width right, width left


def _catmull(p: np.ndarray, sub: int = 8) -> np.ndarray:
    n, t = len(p), np.linspace(0, 1, sub, endpoint=False)[:, None]
    out = []
    for i in range(n):
        p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    return np.concatenate(out)


def convert(name: str, text: str) -> dict:
    raw = parse(text)
    dense = _catmull(raw)  # x, y and both widths through the same spline
    seg = np.linalg.norm(np.roll(dense[:, :2], -1, 0) - dense[:, :2], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(s[-1])
    n = int(round(length))
    q = np.linspace(0, length, n, endpoint=False)
    closed = np.vstack([dense, dense[:1]])
    x, y, wr, wl = (np.interp(q, s, closed[:, j]) for j in range(4))
    p = np.stack([x, y], 1)
    k = _curvature(p, 3)
    folded = int(np.sum(np.abs(k) * np.where(k > 0, wl, wr) >= 1))
    gate_count = max(1, round(length / 40.0))
    r = lambda v: [round(float(a), 3) for a in v]  # noqa: E731
    body = {
        "schema": 1, "circuitId": name.lower(), "revision": 1,
        "source": {"kind": "tumftm", "file": f"{name}.csv", "url": URL.format(name), "license": "LGPL-3.0",
                   "attribution": "TUMFTM racetrack-database (Technical University of Munich, Institute of Automotive Technology)"},
        "officialLengthM": round(length), "lengthM": round(length, 3), "spacingM": round(length / n, 3),
        "centerline": r(p.ravel()), "widthLeft": r(wl), "widthRight": r(wr),
        "gates": [round(i * n / gate_count) for i in range(gate_count)],
        "bounds": {"minX": round(float(x.min() - 15), 3), "minY": round(float(y.min() - 15), 3),
                   "maxX": round(float(x.max() + 15), 3), "maxY": round(float(y.max() + 15), 3)},
        "foldedSamples": folded,
    }
    body["sha256"] = hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()
    return body


def load(name: str) -> fevo.Track:
    """A real circuit as an SDK Track: fevo's shipped ones by id, the converted TUM ones from tracks/real."""
    if name in fevo.track_ids():
        return fevo.load_track(name)
    return fevo.data.Track(json.loads((OUT / f"{name}.track.json").read_text()))


def names(split: str = "all") -> list[str]:
    have = sorted({p.name.removesuffix(".track.json") for p in OUT.glob("*.track.json")} | set(fevo.track_ids()))
    if split == "train":
        return [n for n in have if n not in HOLDOUT and n not in EXCLUDED]
    if split == "test":
        return [n for n in have if n in HOLDOUT]
    return have


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(RAW.glob("*.csv")):
        t = convert(f.stem, f.read_text())
        (OUT / f"{t['circuitId']}.track.json").write_text(json.dumps(t) + "\n")
        w = np.array(t["widthLeft"]) + np.array(t["widthRight"])
        print(f"{t['circuitId']:<14} {t['lengthM'] / 1000:5.2f} km  width {w.min():4.1f}-{w.max():4.1f} m  "
              f"{len(t['gates']):3d} gates  folded samples {t['foldedSamples']}{'  HOLDOUT' if t['circuitId'] in HOLDOUT else ''}")
    print(f"train: {names('train')}\ntest:  {names('test')}")


if __name__ == "__main__":
    main()
