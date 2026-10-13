"""Lap-time tuning with CMA-ES on a NEAT genome's weights.

NEAT is good at finding a structure that gets round. It's slow at fine-tuning:
each mutation nudges one weight at random. This stage keeps the champion's
genome, adds a layer of hidden tanh nodes (connected with zero output weights,
so the car starts out driving exactly like the champion), and lets CMA-ES move
every weight and bias together. The result is still a NEAT genome file that
uploads as one; fitness is the same official-laps-plus-practice score as
lap-time mode.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from typing import Any

import warnings

warnings.filterwarnings("ignore", message="Could not import matplotlib")
import cma  # noqa: E402
import numpy as np

import fevo

from .fitness import _laptime
from .paths import winner_path

_SIM: fevo.Simulator | None = None
HIDDEN_ID0 = 100_000


def _sim() -> fevo.Simulator:
    global _SIM
    if _SIM is None:
        _SIM = fevo.Simulator()
    return _SIM


class Template:
    """A genome with every tunable number addressed by an index into one vector."""

    def __init__(self, genome: dict[str, Any], hidden: int, seed: int, first_id: int = HIDDEN_ID0) -> None:
        rng = np.random.default_rng(seed)
        g = json.loads(json.dumps(genome))
        inputs = g.get("inputKeys") or [-(i + 1) for i in range(15)]
        outputs = g.get("outputKeys") or [0, 1]
        # Keep only expressed connections; NEAT's disabled ones play no part.
        g["connections"] = [c for c in g["connections"] if c.get("enabled", True)]
        have = {(c["in"], c["out"]) for c in g["connections"]}
        # Direct input → output links for every input (zero weight if new).
        for i in inputs:
            for o in outputs:
                if (i, o) not in have:
                    g["connections"].append({"in": i, "out": o, "weight": 0.0, "enabled": True})
        # A fresh hidden layer: small random input weights, zero output weights.
        for h in range(hidden):
            hid = first_id + h
            g["nodes"].append({"id": hid, "type": "hidden", "bias": 0.0, "response": 1.0, "activation": "tanh", "aggregation": "sum"})
            for i in inputs:
                g["connections"].append({"in": i, "out": hid, "weight": float(rng.normal(0, 0.5)), "enabled": True})
            for o in outputs:
                g["connections"].append({"in": hid, "out": o, "weight": 0.0, "enabled": True})
        self.genome = g
        self.slots: list[tuple[str, int]] = [("w", i) for i in range(len(g["connections"]))]
        self.slots += [("b", i) for i, n in enumerate(g["nodes"]) if n.get("type") != "input"]

    def vector(self) -> np.ndarray:
        return np.array([self._get(s) for s in self.slots], dtype=float)

    def _get(self, s: tuple[str, int]) -> float:
        kind, i = s
        return float(self.genome["connections"][i]["weight"] if kind == "w" else self.genome["nodes"][i]["bias"])

    def build(self, x: np.ndarray) -> dict[str, Any]:
        g = json.loads(json.dumps(self.genome))
        for (kind, i), v in zip(self.slots, x):
            v = float(np.clip(v, -30, 30))
            if kind == "w":
                g["connections"][i]["weight"] = v
            else:
                g["nodes"][i]["bias"] = v
        return g


def _score(args: tuple[dict[str, Any], str]) -> tuple[float, dict[str, Any]]:
    genome, track_id = args
    try:
        program = fevo.from_json(json.dumps(genome))
    except fevo.AgentError:
        return -1.0, {}
    sim = _sim()
    track = sim.load_track(track_id)
    return _laptime(sim, program, track)


def tune(track: str, generations: int, hidden: int, sigma: float, popsize: int | None, workers: int | None, seed: int) -> None:
    # Continue from the last tuned car if there is one; each round adds fresh hidden nodes.
    src = next(p for p in (winner_path("tuned", track), winner_path("laptime", track), winner_path("curriculum", track)) if p.is_file())
    print(f"starting from {src.name}", flush=True)
    base = json.loads(src.read_text())
    first_new = HIDDEN_ID0 + max([n["id"] - HIDDEN_ID0 + 1 for n in base["nodes"] if n["id"] >= HIDDEN_ID0] or [0])
    tpl = Template(base, hidden, seed, first_new)
    x0 = tpl.vector()
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    es = cma.CMAEvolutionStrategy(x0, sigma, {"popsize": popsize or 4 + int(3 * np.log(len(x0))) * 2, "seed": seed, "verbose": -9})
    pool = mp.get_context("spawn").Pool(workers)
    best_fit, best_info = _score((tpl.build(x0), track))
    best_x = x0
    print(f"tune {track}: {len(x0)} parameters ({hidden} new hidden nodes), popsize {es.popsize}, start {fevo.format_lap_time(best_info['median_steps'])}", flush=True)
    t0 = time.time()
    out = winner_path("tuned", track)
    try:
        for gen in range(generations):
            xs = es.ask()
            results = pool.map(_score, [(tpl.build(x), track) for x in xs])
            es.tell(xs, [-f for f, _ in results])
            i = int(np.argmax([f for f, _ in results]))
            fit, info = results[i]
            if fit > best_fit:
                best_fit, best_info, best_x = fit, info, xs[i]
                out.write_text(json.dumps(tpl.build(best_x), indent=1) + "\n")
            med = best_info.get("median_steps")
            print(
                f"gen {gen:4d}  {(time.time() - t0) / 60:5.1f} min  best {fevo.format_lap_time(med) if med else '—':>8}  "
                f"practice {best_info.get('practice_m', 0):6.0f} m  this gen {fevo.format_lap_time(info['median_steps']) if info.get('median_steps') else '—':>8}  sigma {es.sigma:.3f}",
                flush=True,
            )
    finally:
        pool.close()
        pool.join()
    out.write_text(json.dumps(tpl.build(best_x), indent=1) + "\n")
    print(f"best {fevo.format_lap_time(best_info['median_steps'])} written to {out}")
