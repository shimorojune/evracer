"""Command-line entry points (see pyproject.toml)."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import fevo

from .paths import WINNERS, winner_path


def _train(mode: str) -> None:
    ap = argparse.ArgumentParser(prog=f"train-{mode}")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("-g", "--generations", type=int, default=1000)
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--resume", help="'latest' (default) or a checkpoint path")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints and start a new population")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from .train import run

    run(mode, a.track, a.generations, a.workers, a.resume, a.fresh, a.seed)


def train_curriculum() -> None:
    _train("curriculum")


def train_laptime() -> None:
    _train("laptime")


def tune_laptime() -> None:
    ap = argparse.ArgumentParser(prog="tune-laptime", description="CMA-ES weight tuning of the lap-time champion")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("-g", "--generations", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=6, help="hidden tanh nodes to add")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--popsize", type=int)
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from .tune import tune

    tune(a.track, a.generations, a.hidden, a.sigma, a.popsize, a.workers, a.seed)


def train_ppo() -> None:
    ap = argparse.ArgumentParser(prog="train-ppo", description="PPO with a reward that pays for daring")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("--steps", type=float, default=50e6, help="total environment steps")
    ap.add_argument("-w", "--workers", type=int)
    ap.add_argument("--envs-per-worker", type=int, default=16)
    ap.add_argument("--daring", type=float, default=1.0, help="progress multiplier per unit of grip usage squared")
    ap.add_argument("--commit", type=float, default=0.5, help="progress multiplier per unit of speed / reference")
    ap.add_argument("--crash", type=float, default=-20.0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from .ppo import Daring, train

    train(a.track, int(a.steps), a.workers, a.envs_per_worker, Daring(daring=a.daring, commit=a.commit, crash=a.crash), a.seed, not a.fresh)


def _winner(track: str, path: str | None) -> Path:
    if path:
        return Path(path)
    for mode in ("ppo", "tuned", "laptime", "curriculum"):
        p = winner_path(mode, track)
        if p.is_file():
            return p
    raise SystemExit(f"No winner for {track} yet; train first.")


def benchmark() -> None:
    ap = argparse.ArgumentParser(prog="benchmark-lap")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("--winner")
    a = ap.parse_args()
    path = _winner(a.track, a.winner)
    program = fevo.from_json(path.read_text())
    res = fevo.evaluate(program, a.track)
    print(f"{path.name} on {a.track} (official rules, same sim.wasm as the website), fingerprint {program.fingerprint()[:6]}")
    for lap in res.laps:
        t = fevo.format_lap_time(lap.steps) if lap.finished else f"{lap.status} at gate {lap.gates_passed} of {lap.gate_count}"
        print(f"  seed {lap.seed + 1}: {t}")
    print(f"  median: {res.lap_time or 'not ranked'}")


def export() -> None:
    ap = argparse.ArgumentParser(prog="export-winner")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    src = _winner(a.track, None)
    out = Path(a.out or f"fevo-racer1-{a.track}.json")
    out.write_text(src.read_text())
    print(f"wrote {out}: upload it on the website. `fevo check {out} --track {a.track}` shows the time it will get.")


def replay() -> None:
    ap = argparse.ArgumentParser(prog="replay")
    ap.add_argument("files", nargs="*", help="agent JSON files (default: every winner in winners/)")
    ap.add_argument("--track", action="append", choices=fevo.track_ids())
    ap.add_argument("-o", "--out", default="replay.html")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    files = [Path(f) for f in a.files] or sorted(WINNERS.glob("*.json"))
    agents = [(f.name, fevo.from_json(f.read_text())) for f in files]
    out = fevo.write_replay(agents, a.out, a.track)
    print(f"wrote {out}")
    if not a.no_open:
        webbrowser.open(out.resolve().as_uri())
