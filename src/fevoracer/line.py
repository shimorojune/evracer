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
"""

from __future__ import annotations

import numpy as np

import fevo


def normals(track: fevo.Track) -> tuple[np.ndarray, np.ndarray]:
    c = np.array(track.raw["centerline"]).reshape(-1, 2)
    t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    return c, np.stack([-t[:, 1], t[:, 0]], 1)


def racing_line(track: fevo.Track, margin: float, step: int = 4, iterations: int = 20000) -> np.ndarray:
    """Offsets from the centre line (metres, left positive) at every centre-line point."""
    c, nl = normals(track)
    n = len(c)
    m = n // step
    idx = np.round(np.arange(m) * n / m).astype(int) % n
    cc, nn = c[idx], nl[idx]
    lo = -(np.array(track.raw["widthRight"])[idx] - margin)
    hi = np.array(track.raw["widthLeft"])[idx] - margin
    o = np.zeros(m)
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
    for j in range(m):
        a = idx[j]
        for s in range(span[j]):
            u = s / span[j]
            i = (a + s) % n
            off[i] = min(wl[i] - margin, max(-(wr[i] - margin), o[j] * (1 - u) + o[(j + 1) % m] * u))
    return off
