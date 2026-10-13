"""Warm start for PPO: teach the network to copy the scripted expert first.

Behaviour cloning with DAgger: the expert drives and the network learns its
(steer, throttle) from the same 15 inputs the network will see; then the
network drives, the expert labels every state it visits (including the
mistakes it would never make itself), and the network trains on everything
so far. A few rounds of that and the network laps on its own. PPO with the
daring reward then starts from this network instead of from scratch.
"""

from __future__ import annotations

import json
import random
import time

import numpy as np
import torch

import fevo

from .expert import Expert
from .paths import CHECKPOINTS, winner_path
from .ppo import LADDER, Policy, mlp_to_neat


def _rollout(sim: fevo.Simulator, expert: Expert, policy: Policy | None, start: dict, steps: int, beta: float, rng: random.Random):
    """Drive one episode; the driver is the expert with probability beta per step, else the policy. Returns (obs, expert actions)."""
    if "seed" in start:
        obs = sim.reset(start["seed"])
    else:
        obs = sim.reset(1, point=start["point"], lateral_m=start["lat"], heading_rad=start["hdg"], speed=start["speed"])
    idx, vx = sim.progress_index, sim.pose()[3]
    xs, ys = [], []
    for _ in range(steps):
        a_exp = expert.act(obs, idx, vx)
        xs.append(list(obs))
        ys.append(a_exp)
        if policy is not None and rng.random() >= beta:
            with torch.no_grad():
                a = policy.pi(torch.tensor([obs], dtype=torch.float32))[0].clamp(-1, 1).tolist()
        else:
            a = a_exp
        code, vals = sim.step_fast(a[0], a[1])
        obs, vx, idx = list(vals[:15]), vals[15], int(vals[18])
        if code != 0:
            break
    return xs, ys


def _starts(track: fevo.Track, expert: Expert, k: int, rng: random.Random) -> list[dict]:
    # The standing start is where imitation is hardest, so drive it several times per round.
    starts: list[dict] = [{"seed": s % 3} for s in range(9)]
    for _ in range(k):
        p = rng.randrange(track.points)
        starts.append({"point": p, "lat": rng.uniform(-2.5, 2.5), "hdg": rng.uniform(-0.08, 0.08), "speed": rng.uniform(0.4, 1.1) * expert.v[p]})
    return starts


def _fit(policy: Policy, X: np.ndarray, Y: np.ndarray, epochs: int) -> float:
    opt = torch.optim.Adam(policy.pi.parameters(), lr=1e-3)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 4096):
            i = perm[s : s + 4096]
            loss = (policy.pi(Xt[i]) - Yt[i]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss)


def warm_start(track_id: str, rounds: int, grip_margin: float, seed: int) -> None:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    sim = fevo.Simulator()
    track = sim.load_track(track_id)
    expert = Expert(track, sim.rules, grip_margin)
    policy = Policy()
    X: list[list[float]] = []
    Y: list[tuple[float, float]] = []
    t0 = time.time()
    best = None
    for r in range(rounds + 1):
        beta = 1.0 if r == 0 else max(0.0, 0.5 - 0.15 * r)
        sim.configure(fevo.Practice(max_steps=12000))
        sim.load_track(track)
        for st in _starts(track, expert, 300, rng):
            xs, ys = _rollout(sim, expert, None if r == 0 else policy, st, 12000 if "seed" in st else 900, beta, rng)
            X += xs
            Y += ys
        loss = _fit(policy, np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), 20 if r == 0 else 10)
        mlp = policy.to_mlp()
        res = fevo.evaluate(fevo.from_json(json.dumps(mlp)), track_id, sim=sim)
        laps = res.lap_time or ", ".join(f"{l.status} at gate {l.gates_passed}" for l in res.laps)
        print(f"round {r}: {len(X):,} labelled states, loss {loss:.4f}, official {laps}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if res.ranked and (best is None or res.median_steps < best):
            best = res.median_steps
            winner_path("warmstart", track_id).write_text(json.dumps(mlp) + "\n")
            winner_path("warmstart", track_id, ".neat.json").write_text(json.dumps(mlp_to_neat(mlp)) + "\n")
    # Hand over to PPO: modest exploration noise, value function trained first.
    with torch.no_grad():
        policy.log_std.fill_(-2.3)  # small noise: big noise on a fine-tuned driver just crashes it
    np.savez_compressed(CHECKPOINTS / track_id / "expert-demos.npz", X=np.array(X, dtype=np.float32), Y=np.array(Y, dtype=np.float32))
    ckpt = CHECKPOINTS / track_id / "ppo.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "opt": None, "steps": 0, "best_lap": best, "stage": len(LADDER) - 1, "critic_warmup": 20}, ckpt)
    print(f"saved warm start to {ckpt} (best official {fevo.format_lap_time(best) if best else 'none'}); now run: uv run train-ppo")

