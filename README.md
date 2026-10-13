# fevo-racer1

NEAT agents that learn real Formula 1 circuits on the fevo simulator, the same `sim.wasm` the fevo website scores with.

Setup (needs the fevo monorepo next door at `../fevo`):

```bash
uv sync
```

## Train

**Curriculum** (learn the whole circuit at low speed, then raise the limit):

```bash
uv run train-curriculum                      # Spielberg, resumes from the latest checkpoint
uv run train-curriculum --track monza --fresh
```

**Lap time** (official rules, three seeds, faster is better). Starts from the curriculum population and uses `config/neat-laptime.ini`, which mutates in small steps to fine-tune a car that already laps:

```bash
uv run train-laptime
```

## Check, export, watch

```bash
uv run benchmark-lap                         # official result, as the website will show it
uv run export-winner -o my-agent.json        # the upload file
uv run replay                                # replay of every winner, opens in the browser
```

Champions are written to `winners/` (JSON upload files plus neat-python pickles); checkpoints to `checkpoints/<track>/`.

## Notes from the first runs

- Every node uses `tanh`. When the throttle output was allowed to mutate to `sigmoid` (which can't go negative) the car could never brake and evolution settled on coasting round at 1% throttle.
- Each curriculum stage has a time budget just above its pace target (60% of the speed limit), so crawling round doesn't pay.
