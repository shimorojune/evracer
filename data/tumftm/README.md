# TUM racetrack-database circuits

`raw/` holds centre-line CSVs (x, y, right width, left width) downloaded from
https://github.com/TUMFTM/racetrack-database (`tracks/<Name>.csv`), by the
Institute of Automotive Technology, Technical University of Munich, licensed
under LGPL-3.0. Downloaded 2026-10-14 for fevo-racer1 training and testing
only; they are not fevo circuits.

`uv run python -m fevoracer.realtracks` converts them into the simulator's
track format under `tracks/real/` (each file records its source URL and
licence). Held out from training: Monza, Austin, Montreal, Spa.
