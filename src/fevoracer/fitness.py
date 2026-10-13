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


def _laptime(sim: fevo.Simulator, program: fevo.Program, track: fevo.Track) -> tuple[float, dict[str, Any]]:
    """The official three laps; every step saved is worth one point per lap."""
    result = fevo.evaluate(program, track, sim=sim)
    fitness = 0.0
    for lap in result.laps:
        fitness += 100.0 * lap.gates_passed
        if lap.finished:
            fitness += 30000.0 - lap.steps
    finished = [l for l in result.laps if l.finished]
    info = {
        "coverage": min(l.gates_passed for l in result.laps) / len(track.gates),
        "finished": result.ranked,
        "laps_finished": len(finished),
        "median_steps": result.median_steps,
        "statuses": [l.status for l in result.laps],
    }
    return fitness, info
