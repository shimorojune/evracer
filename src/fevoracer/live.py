"""Live training snapshots for `uv run watch`.

At each evaluation the trainer writes one small JSON file: the latest
official lap (traced), where training episodes have been ending, and the
history so far. The watch page polls that file from a separate process, so
watching costs training nothing beyond one traced lap and a file write.
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Any

import fevo

from .paths import LOGS


def live_path(mode: str, track: str):
    d = LOGS / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{mode}-{track}.json"


class LiveWriter:
    def __init__(self, mode: str, track: str, window: int = 20) -> None:
        self.mode = mode
        self.track = fevo.load_track(track)
        self.path = live_path(mode, track)
        self.ends: deque[list[tuple[str, int]]] = deque(maxlen=window)
        self.history: list[dict[str, Any]] = []
        self.stage: str | None = None

    def add_ends(self, ends: list[tuple[str, int]]) -> None:
        self.ends.append(ends)

    def write(self, sim: fevo.Simulator, steps: int, minutes: float, result: fevo.OfficialResult, best_lap: int | None, reward: float, std: float) -> None:
        gates = self.track.gates
        bucket = lambda idx: max(0, sum(1 for g in gates if g <= idx) - 1)  # noqa: E731
        counts: dict[str, list[int]] = {}
        for batch in self.ends:
            for status, idx in batch:
                if status in ("crashed", "stopped", "error"):
                    counts.setdefault(status, [0] * len(gates))[bucket(idx)] += 1
        lap = sim.run_lap(1, trace=True)
        t = lap.trace or {}
        k = 2  # 30 Hz is plenty to watch
        rnd = lambda xs, q: [round(v * q) / q for v in xs[::k]]  # noqa: E731
        self.history.append({"steps": steps, "minutes": round(minutes, 2), "gates": min(l.gates_passed for l in result.laps),
                             "lap": result.median_steps, "reward": round(reward, 4)})  # fmt: skip
        snap = {
            "mode": self.mode, "track": self.track.id, "steps": steps, "minutes": round(minutes, 2),
            "reward": reward, "std": std, "bestLap": best_lap, "stage": self.stage,
            "official": [{"seed": l.seed, "status": l.status, "steps": l.steps, "gates": l.gates_passed} for l in result.laps],
            "lap": {"status": lap.status, "steps": lap.steps, "gates": lap.gates_passed, "every": k,
                    "x": rnd(t["x"], 100), "y": rnd(t["y"], 100), "yaw": rnd(t["yaw"], 1000), "speed": rnd(t["speed"], 10),
                    "steer": rnd(t["steer"], 100), "throttle": rnd(t["throttle"], 100), "gate": t["gates"][::k]},
            "ends": counts, "history": self.history[-400:],
        }  # fmt: skip
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, separators=(",", ":")))
        os.replace(tmp, self.path)  # atomic: the page never reads half a file
