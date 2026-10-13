from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "neat.ini"
CONFIG_LAPTIME = ROOT / "config" / "neat-laptime.ini"
CHECKPOINTS = ROOT / "checkpoints"
WINNERS = ROOT / "winners"
LOGS = ROOT / "logs"


def track_dir(track: str) -> Path:
    d = CHECKPOINTS / track
    d.mkdir(parents=True, exist_ok=True)
    return d


def winner_path(mode: str, track: str, suffix: str = ".json") -> Path:
    WINNERS.mkdir(parents=True, exist_ok=True)
    return WINNERS / f"{mode}-{track}{suffix}"
