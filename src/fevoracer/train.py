"""NEAT training loop for both modes, parallel over CPU cores."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import Any

import fevo
import neat

from .curriculum import CurriculumState
from .fitness import Settings, evaluate
from .paths import CONFIG, CONFIG_LAPTIME, track_dir, winner_path


def load_config(mode: str = "curriculum") -> neat.Config:
    path = CONFIG_LAPTIME if mode == "laptime" else CONFIG
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))


def latest_checkpoint(track: str, mode: str) -> Path | None:
    files = sorted(track_dir(track).glob(f"{mode}-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    return files[-1] if files else None


def _eval_task(args: tuple[Any, Any, Settings]) -> tuple[float, dict[str, Any]]:
    genome, config, settings = args
    return evaluate(genome, config, settings)


class Trainer:
    def __init__(self, mode: str, track: str, workers: int) -> None:
        self.mode = mode
        self.track = track
        self.pool = mp.get_context("spawn").Pool(workers)
        self.curriculum = CurriculumState.load(track)
        self.generation = 0
        self.best_ever = float("-inf")
        self.started = time.time()
        self.last_best: tuple[float, dict[str, Any]] | None = None

    def settings(self) -> Settings:
        limit = self.curriculum.speed_limit if self.mode == "curriculum" else None
        return Settings(self.mode, self.track, limit, self.generation)

    def eval_genomes(self, genomes: list[tuple[int, Any]], config: neat.Config) -> None:
        s = self.settings()
        results = self.pool.map(_eval_task, [(g, config, s) for _, g in genomes], chunksize=4)
        best: tuple[float, dict[str, Any], Any] | None = None
        for (_, g), (fit, info) in zip(genomes, results):
            g.fitness = fit
            if best is None or fit > best[0]:
                best = (fit, info, g)
        assert best is not None
        fit, info, champ = best
        self.report(fit, info)
        if fit > self.best_ever:
            self.best_ever = fit
            self.save_champion(champ, config, info)
        if self.mode == "curriculum":
            passed = self.curriculum.passed(info["finished"], info.get("steps", 0), fevo.load_track(self.track).length_m)
            msg = self.curriculum.advance(info["coverage"], passed)
            if msg:
                print(f"  >> {msg}", flush=True)
                self.best_ever = float("-inf")  # fitness scale changes with the limit
                self.save_champion(champ, config, info, stage=True)
        self.generation += 1

    def report(self, fit: float, info: dict[str, Any]) -> None:
        mins = (time.time() - self.started) / 60
        if self.mode == "curriculum":
            extra = f"limit {self.curriculum.label():>17}  coverage {info['coverage'] * 100:5.1f}%"
            if info["finished"]:
                avg = fevo.load_track(self.track).length_m / (info["steps"] / 60) * 3.6
                extra += f"  lap {fevo.format_lap_time(info['steps'])} ({avg:.0f} km/h avg)"
        else:
            med = info.get("median_steps")
            extra = f"laps finished {info['laps_finished']}/3  median {fevo.format_lap_time(med) if med else '—':>8}  {','.join(info['statuses'])}"
        print(f"gen {self.generation:4d}  {mins:6.1f} min  best {fit:10.1f}  {extra}", flush=True)

    def save_champion(self, genome: Any, config: neat.Config, info: dict[str, Any], stage: bool = False) -> None:
        winner_path(self.mode, self.track, ".pkl").write_bytes(pickle.dumps(genome))
        winner_path(self.mode, self.track).write_text(json.dumps(fevo.neat_to_json(genome, config), indent=1) + "\n")
        if stage:
            name = f"{self.mode}-{self.track}-stage{self.curriculum.step:02d}"
            (winner_path(self.mode, self.track).parent / f"{name}.json").write_text(json.dumps(fevo.neat_to_json(genome, config)) + "\n")

    def close(self) -> None:
        self.pool.close()
        self.pool.join()


def run(mode: str, track: str, generations: int, workers: int | None, resume: str | None, fresh: bool, seed: int) -> None:
    config = load_config(mode)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if fresh and mode == "curriculum":
        CurriculumState(track=track).save()

    ckpt: Path | None = None
    if resume == "latest" or (resume is None and not fresh):
        ckpt = latest_checkpoint(track, mode)
        if ckpt is None and mode == "laptime":
            ckpt = latest_checkpoint(track, "curriculum")
            if ckpt:
                print(f"Starting lap-time training from the curriculum population: {ckpt.name}")
    elif resume:
        ckpt = Path(resume)

    if ckpt:
        population = neat.Checkpointer.restore_checkpoint(str(ckpt), new_config=config)
        print(f"Resumed {ckpt.name}")
    else:
        population = neat.Population(config, seed=seed)

    trainer = Trainer(mode, track, workers)
    trainer.generation = population.generation
    population.add_reporter(neat.Checkpointer(10, filename_prefix=str(track_dir(track) / f"{mode}-")))
    print(
        f"fevo-racer1 {mode} on {fevo.load_track(track).name}: population {len(population.population)}, {workers} workers, "
        f"sim.wasm {fevo._wasm.wasm_sha256()[:12]}"
        + (f", curriculum stage {trainer.curriculum.step} ({trainer.curriculum.label()})" if mode == "curriculum" else ""),
        flush=True,
    )
    try:
        for _ in range(generations):
            population.run(trainer.eval_genomes, 1)
            if mode == "curriculum" and trainer.curriculum.graduated:
                print("Curriculum complete. Next: train-laptime", flush=True)
                break
    finally:
        trainer.close()
