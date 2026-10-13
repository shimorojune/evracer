"""PPO on the fevo simulator, with a reward that pays for daring.

The policy sees exactly the 15 official inputs and outputs steer and
throttle, so the trained network uploads as-is (as an MLP, or converted to
a NEAT genome). Episodes start all round the lap at random speeds (practice
settings), plus some from the official standing start.

Reward per step (only ever used for training; official times are lap time):
  metres gained along the track × (1 + daring × grip usage² + commit × speed / reference)

  grip usage is how close the car runs to its cornering limit, so a metre
  taken on the edge pays much more than a metre taken safely; speed relative
  to the racing-line reference pays for carrying speed into corners. Both are
  multipliers on progress, so they can only be earned going forwards (a first
  version paid them per step, and the car learned to do donuts).
  crash       a modest penalty: enough that it learns to brake (at -5 it
              ran into turn 1 at 250 km/h every time), small next to what a
              lap of daring progress earns
  lap         a bonus for finishing, bigger for faster laps
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import fevo

from .paths import CHECKPOINTS, winner_path


@dataclass
class Daring:
    """Reward weights. Raise `daring` and `commit` for a braver car."""

    progress: float = 0.1  # per metre
    commit: float = 0.5  # progress multiplier per unit of speed / reference speed
    daring: float = 1.0  # progress multiplier per unit of grip usage squared
    crash: float = -20.0
    lap: float = 50.0
    lap_time: float = 0.02  # per step under the lap's step limit


# --------------------------------------------------------------------------- curriculum

#: Same ladder as NEAT curriculum mode: m/s, None = the real car.
LADDER: list[float | None] = [14, 18, 23, 29, 36, 44, 53, 63, 75, None]
PACE = 0.6


def stage_budget(length_m: float, limit: float | None, official_max: int) -> int:
    """Lap step limit for a stage: just above what the pace target needs, official limit with no speed limit."""
    if limit is None:
        return official_max
    return int(1.15 * 60 * length_m / (PACE * limit))


def stage_passed(result: fevo.OfficialResult, length_m: float, limit: float | None) -> bool:
    if not all(l.finished for l in result.laps):
        return False
    if limit is None:
        return True
    return length_m / (max(l.steps for l in result.laps) / 60) >= PACE * limit


# --------------------------------------------------------------------------- environments


class RaceEnv:
    """One simulator with random practice starts and the daring reward."""

    def __init__(self, track: str, daring: Daring, max_steps: int, official_start_p: float, rng: random.Random) -> None:
        self.sim = fevo.Simulator()
        self.track = self.sim.load_track(track)
        self.rules = self.sim.rules
        self.ref_speed = fevo.reference(track)["speed"]
        car = self.rules.car
        self.car = car
        self.d = daring
        self.max_steps = max_steps
        self.official_p = official_start_p
        self.rng = rng
        self.n = self.track.points
        self.spacing = self.track.raw["spacingM"]
        self.obs: list[float] = []
        self.idx = 0
        self.t = 0
        self.official_max = self.rules.raw["timing"]["maxSteps"]
        self.lap_limit = self.official_max
        self.official = False
        self.speed_limit: float | None = None
        #: Trouble spots (centre-line indices where training crashes most), sent by the trainer.
        self.hot: list[int] = []

    def set_limit(self, limit: float | None) -> None:
        """Curriculum stage: practice speed limit (None = the real car) and the matching lap time budget."""
        self.speed_limit = limit
        self.lap_limit = stage_budget(self.track.length_m, limit, self.official_max)

    def ref_at(self, idx: int) -> float:
        """Reference speed here, capped by the stage's speed limit."""
        r = self.ref_speed[idx]
        return r if self.speed_limit is None else min(r, self.speed_limit)

    def grip_g(self, v: float) -> float:
        c = self.car
        df = 0.5 * c["airDensityKgM3"] * c["downforceAreaM2"] * v * v
        return 0.5 * (c["tyreMuFront"] + c["tyreMuRear"]) * (1 + df / (c["massKg"] * c["gravityMps2"]))

    def reset(self) -> list[float]:
        self.official = self.rng.random() < self.official_p
        if self.official:
            self.sim.configure(fevo.Practice(speed_limit=self.speed_limit, max_steps=self.lap_limit))
            obs = self.sim.reset(self.rng.randrange(3))
        else:
            self.sim.configure(fevo.Practice(speed_limit=self.speed_limit, max_steps=self.max_steps))
            if self.hot and self.rng.random() < 0.5:
                # Failure practice: start 20–250 m before a spot where cars keep crashing.
                p = (self.rng.choice(self.hot) - int(self.rng.uniform(20, 250) / self.spacing)) % self.n
            else:
                p = self.rng.randrange(self.n)
            # Some starts are faster than the reference allows, so the car has to learn to brake hard.
            v = self.rng.uniform(0.4, 1.25) * self.ref_at(p)
            obs = self.sim.reset(1, point=p, lateral_m=self.rng.uniform(-2, 2), heading_rad=self.rng.uniform(-0.05, 0.05), speed=v)
        self.idx = self.sim.progress_index
        self.t = 0
        self.obs = obs
        return obs

    def step(self, steer: float, throttle: float) -> tuple[list[float], float, bool, dict[str, Any]]:
        code, vals = self.sim.step_fast(steer, throttle)
        status = fevo.sim.STATUS[code]
        obs = list(vals[:15])
        vx, _vy, yaw_rate, idx = vals[15], vals[16], vals[17], int(vals[18])
        steps = int(vals[20])
        d = self.d
        moved = ((idx - self.idx + self.n // 2) % self.n - self.n // 2) * self.spacing
        self.idx = idx
        ref = max(5.0, self.ref_at(idx))
        # Both capped at 1: pay for reaching the limit and the reference speed,
        # not for going past them (that's sliding off or missing the braking
        # point, and an uncapped version taught the car never to brake).
        usage = min(1.0, abs(vx * yaw_rate) / 9.80665 / self.grip_g(vx))
        bonus = 1.0 + d.daring * usage * usage + d.commit * min(1.0, vx / ref)
        r = d.progress * moved * (bonus if moved > 0 else 1.0)
        done = status != "running"
        info: dict[str, Any] = {"status": status}
        if status == "finished":
            r += d.lap + d.lap_time * max(0, self.lap_limit - steps)
            if self.official:  # a full lap from the start line, not a practice run reaching it
                info["lap_steps"] = steps
        elif status in ("crashed", "stopped", "error"):
            r += d.crash
        self.obs = obs
        return obs, r, done, info


def _worker(conn: Any, track: str, daring: dict[str, float], k: int, seed: int, max_steps: int, official_p: float) -> None:
    rng = random.Random(seed)
    envs = [RaceEnv(track, Daring(**daring), max_steps, official_p, rng) for _ in range(k)]
    obs = [e.reset() for e in envs]
    conn.send(np.array(obs, dtype=np.float32))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        if isinstance(msg, tuple) and msg[0] == "hot":
            for e in envs:
                e.hot = msg[1]
            conn.send(True)
            continue
        if isinstance(msg, tuple) and msg[0] == "limit":
            for e in envs:
                e.set_limit(msg[1])
            obs = [e.reset() for e in envs]
            conn.send(np.array(obs, dtype=np.float32))
            continue
        actions = msg
        out_obs, rew, done, laps, ends = [], [], [], [], []
        for e, a in zip(envs, actions):
            o, r, dn, info = e.step(float(a[0]), float(a[1]))
            if dn:
                ends.append((info.get("status", ""), e.idx))
                o = e.reset()
            if "lap_steps" in info:
                laps.append(info["lap_steps"])
            out_obs.append(o)
            rew.append(r)
            done.append(dn)
        conn.send((np.array(out_obs, dtype=np.float32), np.array(rew, dtype=np.float32), np.array(done), laps, ends))


class VecEnv:
    def __init__(self, track: str, daring: Daring, workers: int, per_worker: int, seed: int, max_steps: int, official_p: float) -> None:
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, track, asdict(daring), per_worker, seed * 1000 + w, max_steps, official_p), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.obs = np.concatenate([c.recv() for c in self.conns])
        self.per = per_worker

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        for i, c in enumerate(self.conns):
            c.send(actions[i * self.per : (i + 1) * self.per])
        res = [c.recv() for c in self.conns]
        self.obs = np.concatenate([r[0] for r in res])
        laps = [l for r in res for l in r[3]]
        self.ends = [e for r in res for e in r[4]]
        return self.obs, np.concatenate([r[1] for r in res]), np.concatenate([r[2] for r in res]), laps

    def set_hot(self, points: list[int]) -> None:
        for c in self.conns:
            c.send(("hot", points))
        for c in self.conns:
            c.recv()

    def set_limit(self, limit: float | None) -> np.ndarray:
        for c in self.conns:
            c.send(("limit", limit))
        self.obs = np.concatenate([c.recv() for c in self.conns])
        return self.obs

    def close(self) -> None:
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=5)


# --------------------------------------------------------------------------- policy


class Policy(nn.Module):
    """Mean action = MLP(obs) with tanh hidden layers and a linear output: uploads as an MLP."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(15, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.v = nn.Sequential(nn.Linear(15, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        for m in self.pi:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        # A floor on exploration noise: without it the noise collapsed and training fell apart.
        return torch.distributions.Normal(self.pi(obs), self.log_std.clamp(min=-3.0).exp())

    def to_mlp(self) -> dict[str, Any]:
        """The policy's mean as a fevo MLP upload file (deterministic driving)."""
        layers = [m for m in self.pi if isinstance(m, nn.Linear)]
        return {
            "layers": [15] + [m.out_features for m in layers],
            "activation": "tanh",
            "outputActivation": "identity",
            "weights": [m.weight.detach().double().flatten().tolist() for m in layers],
            "biases": [m.bias.detach().double().tolist() for m in layers],
        }


def mlp_to_neat(mlp: dict[str, Any]) -> dict[str, Any]:
    """The same network as a NEAT genome. neat-python's tanh is tanh(2.5·z), so
    tanh nodes get every incoming weight and bias divided by 2.5; outputs use identity."""
    layers, W, B = mlp["layers"], mlp["weights"], mlp["biases"]
    nodes, conns = [], []
    prev = [-(i + 1) for i in range(layers[0])]
    next_id = 2
    for l in range(1, len(layers)):
        last = l == len(layers) - 1
        ids = [0, 1] if last else list(range(next_id, next_id + layers[l]))
        if not last:
            next_id += layers[l]
        scale = 1.0 if last else 1 / 2.5
        for j, nid in enumerate(ids):
            nodes.append({"id": nid, "type": "output" if last else "hidden", "bias": B[l - 1][j] * scale, "response": 1.0,
                          "activation": "identity" if last else "tanh", "aggregation": "sum"})  # fmt: skip
            for i, src in enumerate(prev):
                conns.append({"in": src, "out": nid, "weight": W[l - 1][j * layers[l - 1] + i] * scale, "enabled": True})
        prev = ids
    return {"inputKeys": [-(i + 1) for i in range(layers[0])], "outputKeys": [0, 1], "nodes": nodes, "connections": conns}


# --------------------------------------------------------------------------- training

LR = 3e-4
ENTROPY = 0.0


def train(track: str, total_steps: int, workers: int | None, per_worker: int, daring: Daring, seed: int, resume: bool) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    n_envs = workers * per_worker
    horizon = 256
    gamma, lam, clip, epochs, mb = 0.997, 0.95, 0.2, 5, 4096
    policy = Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    ckpt = CHECKPOINTS / track / "ppo.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done_steps = 0
    best_lap: int | None = None
    stage = 0
    if resume and ckpt.is_file():
        state = torch.load(ckpt, weights_only=False)
        policy.load_state_dict(state["policy"])
        opt.load_state_dict(state["opt"])
        done_steps, best_lap, stage = state["steps"], state.get("best_lap"), state.get("stage", len(LADDER) - 1)
        print(f"resumed at {done_steps:,} steps, best lap {fevo.format_lap_time(best_lap) if best_lap else '—'}")
    env = VecEnv(track, daring, workers, per_worker, seed, max_steps=30 * 60, official_p=0.3)
    env.set_limit(LADDER[stage])
    sim = fevo.Simulator()
    track_len = fevo.load_track(track).length_m
    stage_name = lambda: "no limit" if LADDER[stage] is None else f"{LADDER[stage] * 3.6:.0f} km/h"  # noqa: E731
    from .live import LiveWriter

    live = LiveWriter("ppo", track)
    print(f"PPO on {track}: {n_envs} envs over {workers} workers, reward {asdict(daring)}, curriculum stage {stage_name()}", flush=True)
    t0 = time.time()
    obs = torch.as_tensor(env.obs)
    update = 0
    try:
        while done_steps < total_steps:
            buf_o, buf_a, buf_lp, buf_r, buf_d, buf_v = [], [], [], [], [], []
            laps: list[int] = []
            ends: list[tuple[str, int]] = []
            for _ in range(horizon):
                with torch.no_grad():
                    dist = policy.dist(obs)
                    a = dist.sample()
                    lp = dist.log_prob(a).sum(-1)
                    v = policy.v(obs).squeeze(-1)
                nobs, r, d, lap = env.step(a.clamp(-1, 1).numpy())
                buf_o.append(obs)
                buf_a.append(a)
                buf_lp.append(lp)
                buf_r.append(torch.as_tensor(r))
                buf_d.append(torch.as_tensor(d, dtype=torch.float32))
                buf_v.append(v)
                laps += lap
                ends += env.ends
                obs = torch.as_tensor(nobs)
            with torch.no_grad():
                last_v = policy.v(obs).squeeze(-1)
            adv = torch.zeros(horizon, n_envs)
            gae = torch.zeros(n_envs)
            for t in reversed(range(horizon)):
                nv = last_v if t == horizon - 1 else buf_v[t + 1]
                nonterm = 1.0 - buf_d[t]
                delta = buf_r[t] + gamma * nv * nonterm - buf_v[t]
                gae = delta + gamma * lam * nonterm * gae
                adv[t] = gae
            ret = adv + torch.stack(buf_v)
            O, A, LP = torch.cat(buf_o), torch.cat(buf_a), torch.cat(buf_lp)
            ADV, RET = adv.flatten(), ret.flatten()
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            for _ in range(epochs):
                perm = torch.randperm(len(O))
                for s in range(0, len(O), mb):
                    i = perm[s : s + mb]
                    dist = policy.dist(O[i])
                    ratio = (dist.log_prob(A[i]).sum(-1) - LP[i]).exp()
                    pl = -torch.min(ratio * ADV[i], ratio.clamp(1 - clip, 1 + clip) * ADV[i]).mean()
                    vl = 0.5 * (policy.v(O[i]).squeeze(-1) - RET[i]).pow(2).mean()
                    loss = pl + 0.5 * vl - ENTROPY * dist.entropy().sum(-1).mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    opt.step()
            done_steps += horizon * n_envs
            update += 1
            live.add_ends(ends)
            # Failure practice: crash spots from recent updates, weighted by how often (as repeats).
            crash_pts = [idx for b in live.ends for st, idx in b if st in ("crashed", "stopped")]
            env.set_hot(crash_pts[-2000:])
            for g in opt.param_groups:  # linear learning-rate decay
                g["lr"] = LR * max(0.1, 1 - done_steps / total_steps)

            if update % 5 == 0:
                mlp = policy.to_mlp()
                program = fevo.from_json(json.dumps(mlp))
                limit = LADDER[stage]
                if limit is None:
                    res = fevo.evaluate(program, track, sim=sim)
                else:
                    sim.configure(fevo.Practice(speed_limit=limit, max_steps=stage_budget(track_len, limit, 12000)))
                    sim.load_track(track)
                    sim.load_program(program)
                    res = fevo.summarize_official([sim.run_lap(s) for s in range(3)])
                tag = ""
                if limit is not None and stage_passed(res, track_len, limit):
                    stage += 1
                    obs = torch.as_tensor(env.set_limit(LADDER[stage]))
                    tag = f"  >> stage passed, speed limit now {stage_name()}"
                if limit is None and res.ranked and (best_lap is None or res.median_steps < best_lap):
                    best_lap = res.median_steps
                    winner_path("ppo", track).write_text(json.dumps(mlp) + "\n")
                    winner_path("ppo", track, ".neat.json").write_text(json.dumps(mlp_to_neat(mlp)) + "\n")
                    tag = "  << new best"
                torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(), "steps": done_steps, "best_lap": best_lap, "stage": stage}, ckpt)
                live.stage = stage_name()
                live.write(sim, done_steps, (time.time() - t0) / 60, res, best_lap, float(torch.stack(buf_r).mean()), float(policy.log_std.exp().mean()))
                official = res.lap_time or ",".join(f"{l.status}@{l.gates_passed}" for l in res.laps)
                train_laps = f"{fevo.format_lap_time(min(laps))}" if laps else "—"
                print(
                    f"[{stage_name():>9}] {done_steps / 1e6:7.2f}M steps  {(time.time() - t0) / 60:5.1f} min  reward/step {torch.stack(buf_r).mean():.3f}  "
                    f"std {policy.log_std.exp().mean():.2f}  training laps {len(laps):3d} (best {train_laps})  official {official}{tag}",
                    flush=True,
                )
    finally:
        env.close()
