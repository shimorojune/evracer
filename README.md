# evracer

Trains neural-network drivers (NEAT genomes and MLPs) for real Formula 1 circuits on the fevo simulator, the same `sim.wasm` the fevo dashboard runs. The Python package is `fevoracer`.

**Current state:** training is frozen. The house driver is `winners/house-driver-solo6.json` (rules solo-6). [HANDOFF-training.md](HANDOFF-training.md) has its results, how it was made and what to try if training resumes.

## Setup

Needs the [fevo](https://github.com/shimorojune/fevo) repo checked out next door at `../fevo`, because the `fevo` Python SDK is installed from `../fevo/python/fevo`.

```bash
uv sync
```

## Layout

- `src/fevoracer/`: trainers (`general.py` DAgger, `general_es.py` ES, `practice*.py`, `smooth.py`, `ppo.py`, ...), checks (`robust_check.py`, `hashes.py`, `loophole_scan.py`, ...) and the live training page (`watch.py`).
- `winners/`: saved drivers. Top-level JSON files are used as car setups and starting points, so keep them.
- `config/`: NEAT configs. `data/tumftm/`: real circuit layouts. `lines/`: racing lines.
- Generated output (`checkpoints/`, `logs/`, `tracks/`, `*.html`) is gitignored.

## Early single-track tools

These came before the general driver and train on one track at a time (mostly Spielberg).

**Curriculum** (learn the whole circuit at low speed, then raise the limit):

```bash
uv run train-curriculum                      # Spielberg, resumes from the latest checkpoint
uv run train-curriculum --track monza --fresh
```

**Lap time** (official rules, three seeds, faster is better). Starts from the curriculum population and uses `config/neat-laptime.ini`, which mutates in small steps to fine-tune a car that already laps:

```bash
uv run train-laptime
```

**Tuning** (CMA-ES on the champion's weights; the result is still a NEAT genome). Adds a few hidden tanh nodes with zero output weights, then moves every weight together. Much better than NEAT at fine-tuning a car that already laps:

```bash
uv run tune-laptime                          # continues from winners/tuned-<track>.json if present
```

`src/fevoracer/analysis.py` computes a reference lap (the fastest this car could go round the centre line) and the time lost gate by gate.

## Watch training live

While any trainer runs, in a second terminal:

```bash
uv run watch                                 # opens http://localhost:8765
```

The page shows the latest official lap driven by the current network (looping, with a crash marker where it ends), a map of where training runs have been crashing or stopping, and progress over time. The trainer only writes one small snapshot per evaluation (one extra traced lap, about 0.1 s every few updates); the page runs in its own process, so training speed is unaffected. PPO writes snapshots today.

## Check, export, watch

```bash
uv run benchmark-lap                         # official result, as the dashboard scores it
uv run export-winner -o my-agent.json        # agent JSON (publish to the dashboard with `fevo publish` from the fevo repo)
uv run replay                                # replay of every winner, opens in the browser
```

Champions are written to `winners/` (agent JSON files); checkpoints to `checkpoints/<track>/`.

## Notes from the first runs

- Every node uses `tanh`. When the throttle output was allowed to mutate to `sigmoid` (which can't go negative) the car could never brake and evolution settled on coasting round at 1% throttle.
- Each curriculum stage has a time budget just above its pace target (60% of the speed limit), so crawling round doesn't pay.
- NEAT lap mode stalls quickly: each mutation nudges one weight at random and almost every faster variant crashes somewhere. CMA-ES tuning took Spielberg from 1:38.03 to 1:31.02 in 14 minutes; later rounds plateaued.
- The car still brakes lightly over long distances instead of late and hard, and is slowest relative to the reference in fast corners.
