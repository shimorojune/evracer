"""Genome evaluation. Every lap runs entirely inside fevo's sim.wasm, the same
build the website scores with; only the fitness shaping lives here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fevo

#: Curriculum laps at low speed limits are long: 4.3 km at 50 km/h is 5 minutes.
CURRICULUM_MAX_STEPS = 60 * 60 * 8
#: Practice segments started further round the lap, so later corners train in parallel.
SEGMENT_STEPS = 60 * 40
SEGMENT_STARTS = (1 / 3, 2 / 3)
#: Pace reference once there's no speed limit (m/s, about 324 km/h).
REFERENCE_SPEED = 90.0

_SIM: fevo.Simulator | None = None


def _sim() -> fevo.Simulator:
    global _SIM
    if _SIM is None:
        _SIM = fevo.Simulator()
    return _SIM


@dataclass(frozen=True)
class Settings:
    mode: str  # "curriculum" | "laptime"
    track: str
    speed_limit: float | None
    generation: int


def evaluate(genome: Any, config: Any, s: Settings) -> tuple[float, dict[str, Any]]:
    try:
        program = fevo.from_neat(genome, config)
    except fevo.AgentError as e:
        return -1.0, {"error": str(e), "coverage": 0.0, "finished": False}
    sim = _sim()
    track = sim.load_track(s.track)
    if s.mode == "curriculum":
        return _curriculum(sim, program, track, s)
    return _laptime(sim, program, track)


def _curriculum(sim: fevo.Simulator, program: fevo.Program, track: fevo.Track, s: Settings) -> tuple[float, dict[str, Any]]:
    G = len(track.gates)
    n = track.points
    # 1. From the start line, the seed rotating each generation, under the speed limit.
    # The time allowed is just over what the stage's pace target needs, so a car
    # that crawls runs out of time and every extra km/h reaches more gates.
    budget = time_budget(track.length_m, s.speed_limit)
    sim.configure(fevo.Practice(speed_limit=s.speed_limit, max_steps=budget))
    sim.load_program(program)
    sim.reset(s.generation % 3)
    status = sim.run_agent_lap()
    gates = sim.gates_passed
    finished = status == "finished"
    fitness = 100.0 * gates
    if gates == 0 and sim.progress_index < n // 2:
        fitness += 2.5 * min(40, sim.progress_index)  # the first 40 m, before gate 1
    if finished:
        # Pay for pace, not just for finishing: a crawl that never risks a crash
        # otherwise beats a quick lap that sometimes goes off.
        avg = track.length_m / (sim.steps / 60)
        fitness += 5000.0 + 25000.0 * min(1.0, avg / (s.speed_limit or REFERENCE_SPEED))
    info = {"coverage": 1.0 if finished else gates / G, "finished": finished, "steps": sim.steps, "status": status}

    # 2. Short practice segments from further round, so every corner gets trained.
    sim.configure(fevo.Practice(speed_limit=s.speed_limit, max_steps=SEGMENT_STEPS))
    for frac in SEGMENT_STARTS:
        sim.reset(1, point=round(frac * n))
        g0 = sim.gates_passed
        sim.run_agent_lap()
        fitness += 100.0 * (sim.gates_passed - g0)  # fixed time, so more gates means faster
    return fitness, info


def time_budget(length_m: float, limit: float | None) -> int:
    from .curriculum import OFFICIAL_MAX_STEPS, PACE

    if limit is None:
        return OFFICIAL_MAX_STEPS
    return min(CURRICULUM_MAX_STEPS, int(1.15 * 60 * length_m / (PACE * limit)))


#: At-speed practice for lap-time mode: short runs from points spread round
#: the lap, entered at 80% of the reference speed there, scored by metres
#: covered. Rewards carrying speed through each corner without risking the
#: whole lap on it.
PRACTICE_POINTS = 16
PRACTICE_STEPS = 6 * 60
#: Each practice point is entered at both speeds: at 80% the car learns to
#: carry speed, at 100% it has to brake hard and late to survive.
PRACTICE_ENTRIES = (0.8, 1.0)
PRACTICE_WEIGHT = 3.0

_REF: dict[str, list[float]] = {}


def _reference_speeds(track: fevo.Track) -> list[float]:
    if track.id not in _REF:
        from .analysis import reference_lap

        _REF[track.id] = reference_lap(track).speed
    return _REF[track.id]


def _laptime(sim: fevo.Simulator, program: fevo.Program, track: fevo.Track) -> tuple[float, dict[str, Any]]:
    """The official three laps (every step saved is worth a point per lap), plus at-speed practice."""
    result = fevo.evaluate(program, track, sim=sim)
    fitness = 0.0
    for lap in result.laps:
        fitness += 100.0 * lap.gates_passed
        if lap.finished:
            fitness += 30000.0 - lap.steps

    ref = _reference_speeds(track)
    n = track.points
    sim.configure(fevo.Practice(max_steps=PRACTICE_STEPS))
    practice_m = 0.0
    for k in range(PRACTICE_POINTS):
        start = (k * n) // PRACTICE_POINTS + n // (2 * PRACTICE_POINTS)
        for entry in PRACTICE_ENTRIES:
            sim.reset(1, point=start, speed=entry * ref[start])
            status = sim.run_agent_lap()
            metres = ((sim.progress_index - start) % n) * track.raw["spacingM"]
            if metres > n / 2:  # went backwards
                metres = 0.0
            practice_m += (metres if status in ("timed-out", "finished") else 0.5 * metres) / len(PRACTICE_ENTRIES)
    fitness += PRACTICE_WEIGHT * practice_m
    sim.configure()

    finished = [l for l in result.laps if l.finished]
    info = {
        "coverage": min(l.gates_passed for l in result.laps) / len(track.gates),
        "finished": result.ranked,
        "laps_finished": len(finished),
        "median_steps": result.median_steps,
        "statuses": [l.status for l in result.laps],
        "practice_m": practice_m,
    }
    return fitness, info
