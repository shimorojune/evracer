"""A racing line for any track: tools/calibrate's minimum-curvature line, in numpy.

Same objective as racingLine() in tools/calibrate/src/line.ts: on a coarse
copy of the centre line (every `step` points), minimise the summed squared
second difference of the line's points, each point free to move along the
track normal within `margin` of the edges, then interpolate the offsets back to
every centre-line point. The TypeScript solver sweeps point by point
(Gauss-Seidel); here even and odd points take turns (red-black), which keeps it
vectorised and converges to the same kind of line.

margin is how close the car's centre may come to an edge: 1.45 m for the
solo-1/2 limit with 0.5 m safety, -0.45 m (past the edge) for solo-3/4's
one-tyre limit with the same safety.

Summed squared curvature tolerates one sharp spike: at a hairpin whose inside
edge is almost a point (Austin T1: centre-line radius 10.2 m, 9.4 m of track
inside it) the line hugged the apex at a 7.2 m radius, tighter than the car's
8.5 m turning circle at full lock. So wherever the line's radius falls below
`min_radius` (12 m: that circle plus room for the tyres' slip angles), the
inside limit there moves out by half a metre and the line is solved again,
until no point is too tight. Only the stretches within 150 m of a point that
was too tight change; the rest of the line is the original one.

The limit is off by default (min_radius 0): with it the expert's own laps got no
better (Austin: it laps on the old line and crashes on the limited one; the
pure-pursuit expert already cuts inside a too-tight apex), so experts and
training keep the original line. expert_check compares the two.
"""

from __future__ import annotations

import numpy as np

import fevo


def normals(track: fevo.Track) -> tuple[np.ndarray, np.ndarray]:
    c = np.array(track.raw["centerline"]).reshape(-1, 2)
    t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    return c, np.stack([-t[:, 1], t[:, 0]], 1)


def _curv(p: np.ndarray, h: int = 3) -> np.ndarray:
    d = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    th = np.arctan2(d[:, 1], d[:, 0])
    dth = (np.roll(th, -h) - np.roll(th, h) + np.pi) % (2 * np.pi) - np.pi
    return dth / np.maximum(np.linalg.norm(np.roll(p, -h, 0) - np.roll(p, h, 0), axis=1), 1e-6)


def racing_line(track: fevo.Track, margin: float, step: int = 4, iterations: int = 20000,
                min_radius: float = 0.0, rounds: int = 40) -> np.ndarray:
    """Offsets from the centre line (metres, left positive) at every centre-line point; no radius under min_radius
    where the track leaves room for one (0 turns the limit off, the original line)."""
    off = _solve(track, margin, step, iterations)
    if not min_radius:
        return off
    base = off.copy()
    c, nl = normals(track)
    n = len(c)
    touched = np.zeros(n, bool)
    wl, wr = np.array(track.raw["widthLeft"]), np.array(track.raw["widthRight"])
    lo, hi = -(wr - margin), wl - margin
    for _ in range(rounds):
        k = _curv(c + nl * off[:, None])
        tight = np.where(np.abs(k) * min_radius > 1)[0]
        if len(tight) == 0:
            break
        for i in tight:
            near = (i + np.arange(-6, 7)) % n
            touched[(i + np.arange(-150, 151)) % n] = True
            if k[i] > 0:  # turning left: the inside is the left edge, keep the line further right
                hi[near] = np.maximum(lo[near], np.minimum(hi[near], off[near] - 0.5))
            else:
                lo[near] = np.minimum(hi[near], np.maximum(lo[near], off[near] + 0.5))
        off = _solve(track, margin, step, iterations // 4, bounds=(lo, hi), start=off)
    # Keep the change local: the re-solves also carry on converging everywhere else, which would move the whole
    # line. Use them only within 150 m of a point that was too tight, blended in over 50 m either side.
    w = np.convolve(np.concatenate([touched[-50:], touched, touched[:50]]).astype(float), np.ones(101) / 101, "same")[50:-50]
    w = np.clip(w * 2, 0, 1)
    return base * (1 - w) + off * w


def _solve(track: fevo.Track, margin: float, step: int, iterations: int,
           bounds: tuple[np.ndarray, np.ndarray] | None = None, start: np.ndarray | None = None) -> np.ndarray:
    c, nl = normals(track)
    n = len(c)
    m = n // step
    idx = np.round(np.arange(m) * n / m).astype(int) % n
    cc, nn = c[idx], nl[idx]
    if bounds is None:
        lo = -(np.array(track.raw["widthRight"])[idx] - margin)
        hi = np.array(track.raw["widthLeft"])[idx] - margin
    else:
        lo, hi = bounds[0][idx], bounds[1][idx]
    o = np.zeros(m) if start is None else np.clip(start[idx], lo, hi)
    parity = [np.arange(0, m, 2), np.arange(1, m, 2)]
    for _ in range(iterations):
        for sel in parity:
            p = cc + nn * o[:, None]
            tgt = (4 * (np.roll(p, 1, 0) + np.roll(p, -1, 0)) - (np.roll(p, 2, 0) + np.roll(p, -2, 0))) / 6
            new = np.einsum("ij,ij->i", tgt - cc, nn)
            o[sel] = np.clip(new[sel], lo[sel], hi[sel])
    off = np.empty(n)
    nxt = np.roll(idx, -1)
    span = (nxt - idx) % n
    wl, wr = np.array(track.raw["widthLeft"]), np.array(track.raw["widthRight"])
    if bounds is not None:
        wl, wr = bounds[1] + margin, -bounds[0] + margin
    for j in range(m):
        a = idx[j]
        for s in range(span[j]):
            u = s / span[j]
            i = (a + s) % n
            off[i] = min(wl[i] - margin, max(-(wr[i] - margin), o[j] * (1 - u) + o[(j + 1) % m] * u))
    return off
