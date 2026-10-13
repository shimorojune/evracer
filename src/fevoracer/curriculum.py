"""Curriculum: learn the whole circuit slowly, then raise the speed limit.

Like evracer's rising throttle cap, but using the simulator's practice speed
limit: below it the car drives with full physics; above it throttle is cut.
The champion has to lap from the start line, averaging at least 60% of the
limit, before the limit goes up. Once it laps with no limit at all, the curriculum is done and lap-time
mode takes over.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .paths import track_dir

#: Speed limits in m/s (14 m/s is 50 km/h). None = no limit, the official car.
LADDER: list[float | None] = [14, 18, 23, 29, 36, 44, 53, 63, 75, None]
#: A limited stage is passed by a lap averaging at least this share of the limit,
#: so the car has to use the speed it's given, not just crawl round.
PACE = 0.6
#: The last stage is passed by a lap that would also count officially.
OFFICIAL_MAX_STEPS = 12000


@dataclass
class CurriculumState:
    track: str
    step: int = 0
    best_coverage: float = 0.0
    generations: int = 0
    generations_at_step: int = 0
    graduated: bool = False

    @property
    def speed_limit(self) -> float | None:
        return LADDER[self.step]

    def label(self) -> str:
        lim = self.speed_limit
        return "no limit" if lim is None else f"{lim:.0f} m/s ({lim * 3.6:.0f} km/h)"

    def save(self) -> None:
        (track_dir(self.track) / "curriculum.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, track: str) -> "CurriculumState":
        p = track_dir(track) / "curriculum.json"
        return cls(**json.loads(p.read_text())) if p.is_file() else cls(track=track)

    def passed(self, finished: bool, steps: int, length_m: float) -> bool:
        if not finished:
            return False
        if self.speed_limit is None:
            return steps <= OFFICIAL_MAX_STEPS
        return length_m / (steps / 60) >= PACE * self.speed_limit

    def advance(self, champion_coverage: float, champion_passed: bool) -> str | None:
        """Update after a generation. Returns a message when the stage changes."""
        self.generations += 1
        self.generations_at_step += 1
        self.best_coverage = max(self.best_coverage, champion_coverage)
        msg = None
        if champion_passed:
            if self.speed_limit is None:
                self.graduated = True
                msg = "Champion lapped with no speed limit. Curriculum complete."
            else:
                self.step += 1
                self.generations_at_step = 0
                self.best_coverage = 0.0
                msg = f"Champion lapped at pace. Speed limit raised to {self.label()}."
        self.save()
        return msg
