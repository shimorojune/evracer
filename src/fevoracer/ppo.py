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

import copy
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
    """Reward weights: what the driver is paid for. Training only; official times are lap time."""

    progress: float = 0.1  # per metre gained along the track
    #: Progress multiplier for running at the limit: grip usage (cornering, braking and
    #: drive combined, as a share of what the tyres can give) to the 4th power, so 95%
    #: pays far more than 80%. Late, hard braking and throwing it into corners both count.
    daring: float = 2.0
    #: Progress multiplier for speed against the racing-line reference (the fastest the
    #: car can go there), also to the 4th power: be at the limit, not near it.
    commit: float = 1.0
    #: Per gate: seconds gained (+) or lost (−) against the reference lap over the last 40 m.
    split: float = 5.0
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


# --------------------------------------------------------------------------- setups

KNOBS = fevo.load_rules().setup
N_SETUP = len(KNOBS)
OBS_DIM = 15 + N_SETUP


def setup_norm(values: list[float]) -> list[float]:
    """Setup values scaled to about -1..1 around the default, as the network sees them."""
    return [(v - k["default"]) / ((k["max"] - k["min"]) / 2) for v, k in zip(values, KNOBS)]


def setup_from_norm(x: list[float]) -> list[float]:
    """Inverse of setup_norm, clipped to the legal ranges."""
    return [min(k["max"], max(k["min"], k["default"] + v * (k["max"] - k["min"]) / 2)) for v, k in zip(x, KNOBS)]


def default_setup() -> list[float]:
    return [float(k["default"]) for k in KNOBS]


def setup_dict(values: list[float]) -> dict[str, float]:
    return {k["id"]: round(float(v), 4) for k, v in zip(KNOBS, values)}


# --------------------------------------------------------------------------- environments


class RaceEnv:
    """One simulator with random practice starts and the daring reward."""

    def __init__(self, track: str, daring: Daring, max_steps: int, official_start_p: float, rng: random.Random) -> None:
        self.sim = fevo.Simulator()
        self.track = self.sim.load_track(track)
        self.rules = self.sim.rules
        self.ref_speed = fevo.reference(track)["speed"]
        # Reference time at every centre-line point, for gate-to-gate splits.
        sp = self.track.raw["spacingM"]
        self.ref_t = [0.0]
        for v in self.ref_speed:
            self.ref_t.append(self.ref_t[-1] + sp / max(1.0, v))
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
        #: Practice setups: each episode drives a random legal setup (a quarter of them the default),
        #: and the network sees the setup as extra inputs, so it learns to drive any of them.
        self.random_setups = True
        #: How far from the default setups may go (0 = always default, 1 = the whole legal range).
        #: The trainer widens it while the driver keeps finishing laps.
        self.spread = 0.0
        self.setup = default_setup()
        self.snorm = setup_norm(self.setup)

    def set_limit(self, limit: float | None) -> None:
        """Curriculum stage: practice speed limit (None = the real car) and the matching lap time budget."""
        self.speed_limit = limit
        self.lap_limit = stage_budget(self.track.length_m, limit, self.official_max)

    def ref_at(self, idx: int) -> float:
        """Reference speed here, capped by the stage's speed limit."""
        r = self.ref_speed[idx]
        return r if self.speed_limit is None else min(r, self.speed_limit)

    def grip_g(self, v: float) -> float:
        c = self.car  # the current setup's car
        df = 0.5 * c["airDensityKgM3"] * c["downforceAreaM2"] * v * v
        return 0.5 * (c["tyreMuFront"] + c["tyreMuRear"]) * (1 + df / (c["massKg"] * c["gravityMps2"]))

    def reset(self) -> list[float]:
        if self.random_setups:
            if self.rng.random() < 0.25 or self.spread <= 0:
                self.setup = default_setup()
            else:
                # Within the current spread of the default, rounded to the hundredth like a real setup sheet.
                self.setup = [
                    round(min(k["max"], max(k["min"], k["default"] + self.rng.uniform(-1, 1) * self.spread * (k["max"] - k["min"]) / 2)), 2)
                    for k in KNOBS
                ]
            self.snorm = setup_norm(self.setup)
            self.sim.set_setup(self.setup)
            self.car = self.rules.car_for(self.setup)
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
        self.vx_prev = self.sim.pose()[3]
        self.gates = self.sim.gates_passed
        self.gate_step: int | None = None  # step and point of the last gate crossed this episode
        self.gate_idx = 0
        self.obs = obs + self.snorm
        return self.obs

    def step(self, steer: float, throttle: float) -> tuple[list[float], float, bool, dict[str, Any]]:
        code, vals = self.sim.step_fast(steer, throttle)
        status = fevo.sim.STATUS[code]
        obs = list(vals[:15]) + self.snorm
        vx, _vy, yaw_rate, idx = vals[15], vals[16], vals[17], int(vals[18])
        steps = int(vals[20])
        d = self.d
        moved = ((idx - self.idx + self.n // 2) % self.n - self.n // 2) * self.spacing
        self.idx = idx
        ref = max(5.0, self.ref_at(idx))
        # Grip usage: cornering and braking/drive combined, against what the tyres can give.
        # Both bonuses are capped at 1 (going past the limit is sliding, or missing the
        # braking point; an uncapped version taught the car never to brake), and raised to
        # the 4th power so the last few percent pay the most.
        a_lat = vx * yaw_rate
        a_long = (vx - self.vx_prev) * 60.0
        self.vx_prev = vx
        usage = min(1.0, math.hypot(a_lat, a_long) / 9.80665 / self.grip_g(vx))
        pace = min(1.0, vx / ref)
        bonus = 1.0 + d.daring * usage**4 + d.commit * pace**4
        r = d.progress * moved * (bonus if moved > 0 else 1.0)
        gates = int(vals[19])
        if gates > self.gates:
            # Gate split against the reference lap: seconds gained pay, seconds lost cost.
            if self.gate_step is not None:
                agent_s = (steps - self.gate_step) / 60.0
                ref_s = (self.ref_t[idx] - self.ref_t[self.gate_idx]) % self.ref_t[-1]
                r += d.split * (ref_s - agent_s)
            self.gate_step, self.gate_idx = steps, idx
        self.gates = gates
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
        if isinstance(msg, tuple) and msg[0] == "spread":
            for e in envs:
                e.spread = msg[1]
            conn.send(True)
            continue
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

    def set_spread(self, spread: float) -> None:
        for c in self.conns:
            c.send(("spread", spread))
        for c in self.conns:
            c.recv()

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

    def __init__(self, hidden: int = 64, in_dim: int = OBS_DIM) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.pi = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.v = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        for m in self.pi:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        # A floor on exploration noise: without it the noise collapsed and training fell apart.
        return torch.distributions.Normal(self.pi(obs), self.log_std.clamp(min=-3.0).exp())

    def to_mlp(self, setup: list[float] | None = None) -> dict[str, Any]:
        """The policy's mean as a fevo MLP upload file for one setup.

        The setup inputs are constant for a whole run, so they're folded into
        the first layer's biases: the file is a plain 15-input network plus
        the "setup" block, and drives exactly as the setup-aware network."""
        setup = setup or default_setup()
        layers = [m for m in self.pi if isinstance(m, nn.Linear)]
        w0 = layers[0].weight.detach().double()
        b0 = layers[0].bias.detach().double()
        if self.in_dim > 15:
            b0 = b0 + w0[:, 15:] @ torch.tensor(setup_norm(setup), dtype=torch.float64)
            w0 = w0[:, :15]
        weights = [w0.flatten().tolist()] + [m.weight.detach().double().flatten().tolist() for m in layers[1:]]
        biases = [b0.tolist()] + [m.bias.detach().double().tolist() for m in layers[1:]]
        return {
            "layers": [15] + [m.out_features for m in layers],
            "activation": "tanh",
            "outputActivation": "identity",
            "weights": weights,
            "biases": biases,
            "setup": setup_dict(setup),
        }

    def load_compatible(self, state: dict[str, Any]) -> None:
        """Load a checkpoint, widening a 15-input network to also take setup inputs (new weights start at zero)."""
        own = self.state_dict()
        for key, val in state.items():
            if key in own and own[key].shape != val.shape and val.dim() == 2 and val.shape[0] == own[key].shape[0]:
                widened = torch.zeros_like(own[key])
                widened[:, : val.shape[1]] = val
                state[key] = widened
        self.load_state_dict(state)


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
    out = {"inputKeys": [-(i + 1) for i in range(layers[0])], "outputKeys": [0, 1], "nodes": nodes, "connections": conns}
    if "setup" in mlp:
        out["setup"] = mlp["setup"]
    return out


# --------------------------------------------------------------------------- training

LR = 1e-4
#: Evaluations in a row without a finished official lap before rolling back to the best driver.
ROLLBACK_AFTER = 6
#: After a warm start, keep imitating the expert as well, fading out over this share of training
#: (without it PPO's first updates wrecked the imitated driver within a million steps).
BC_FADE = 0.4
ENTROPY = 0.0


def train(track: str, total_steps: int, workers: int | None, per_worker: int, daring: Daring, seed: int, resume: bool, critic_warmup_override: int = 0) -> None:
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
    critic_warmup = 0
    misses, lr_scale, best_state = 0, 1.0, None
    spread = 0.1
    if resume and ckpt.is_file():
        state = torch.load(ckpt, weights_only=False)
        widened = state["policy"]["pi.0.weight"].shape[1] != policy.in_dim
        policy.load_compatible(state["policy"])
        if state.get("opt") and not widened:
            opt.load_state_dict(state["opt"])
        if widened:
            print(f"widened the network to {policy.in_dim} inputs (setup inputs start at zero)")
            # The starting driver is the fallback until a faster one is found.
            best_state = {"policy": copy.deepcopy(policy.state_dict()), "opt": copy.deepcopy(opt.state_dict())}
        critic_warmup = max(state.get("critic_warmup", 0), critic_warmup_override)
        best_file = ckpt.with_name("ppo-best.pt")
        if best_file.is_file() and not widened:
            best_state = torch.load(best_file, weights_only=False)
        done_steps, best_lap, stage = state["steps"], state.get("best_lap"), state.get("stage", len(LADDER) - 1)
        print(f"resumed at {done_steps:,} steps, best lap {fevo.format_lap_time(best_lap) if best_lap else '—'}")
    demos = CHECKPOINTS / track / "expert-demos.npz"
    bc_x = bc_y = None
    if resume and demos.is_file() and done_steps == 0:
        d = np.load(demos)
        bc_x, bc_y = torch.as_tensor(d["X"]), torch.as_tensor(d["Y"])
        if bc_x.shape[1] < OBS_DIM:  # demos were driven on the default setup: its normalised inputs are 0
            bc_x = torch.cat([bc_x, torch.zeros(len(bc_x), OBS_DIM - bc_x.shape[1])], dim=1)
        print(f"keeping {len(bc_x):,} expert demonstrations in the loss, fading out over {BC_FADE:.0%} of training")
    env = VecEnv(track, daring, workers, per_worker, seed, max_steps=30 * 60, official_p=0.3)
    env.set_limit(LADDER[stage])
    env.set_spread(spread)
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
                    # After a warm start, train only the value function for a few updates,
                    # so untrained value estimates don't wreck the imitated policy.
                    loss = (0.5 * vl) if update < critic_warmup else (pl + 0.5 * vl - ENTROPY * dist.entropy().sum(-1).mean())
                    if bc_x is not None:
                        w = max(0.0, 1.0 - done_steps / (BC_FADE * total_steps))
                        if w > 0:
                            j = torch.randint(0, len(bc_x), (len(i),))
                            loss = loss + 2.0 * w * (policy.pi(bc_x[j]) - bc_y[j]).pow(2).mean()
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
                g["lr"] = LR * lr_scale * max(0.1, 1 - done_steps / total_steps)

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
                if limit is None and not res.ranked:
                    misses += 1
                    if best_state is not None and misses >= ROLLBACK_AFTER:
                        # Collapse guard: roll back to the best driver and slow learning down.
                        policy.load_state_dict(best_state["policy"])
                        opt.load_state_dict(best_state["opt"])
                        lr_scale = max(0.1, lr_scale * 0.5)  # a floor, so it keeps learning after repeated rollbacks
                        misses = 0
                        spread = max(0.1, spread * 0.7)
                        env.set_spread(spread)
                        tag = f"  << rolled back to best ({fevo.format_lap_time(best_lap)}), learning rate x{lr_scale:.2f}"
                else:
                    misses = 0
                if limit is None and res.ranked and spread < 1.0:
                    # Still finishing laps: let practice setups stray further from the default.
                    spread = min(1.0, spread * 1.15)
                    env.set_spread(spread)
                if limit is None and res.ranked and (best_lap is None or res.median_steps < best_lap):
                    best_state = {"policy": copy.deepcopy(policy.state_dict()), "opt": copy.deepcopy(opt.state_dict())}
                    torch.save(best_state, ckpt.with_name("ppo-best.pt"))
                    best_lap = res.median_steps
                    winner_path("ppo", track).write_text(json.dumps(mlp) + "\n")
                    winner_path("ppo", track, ".neat.json").write_text(json.dumps(mlp_to_neat(mlp)) + "\n")
                    tag = "  << new best"
                torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(), "steps": done_steps, "best_lap": best_lap, "stage": stage}, ckpt)
                live.stage = f"{stage_name()}, setups ±{spread:.0%}"
                live.write(sim, done_steps, (time.time() - t0) / 60, res, best_lap, float(torch.stack(buf_r).mean()), float(policy.log_std.exp().mean().detach()))
                official = res.lap_time or ",".join(f"{l.status}@{l.gates_passed}" for l in res.laps)
                train_laps = f"{fevo.format_lap_time(min(laps))}" if laps else "—"
                print(
                    f"[{stage_name():>9} setups ±{spread:4.0%}] {done_steps / 1e6:7.2f}M steps  {(time.time() - t0) / 60:5.1f} min  reward/step {torch.stack(buf_r).mean():.3f}  "
                    f"std {policy.log_std.exp().mean():.2f}  training laps {len(laps):3d} (best {train_laps})  official {official}{tag}",
                    flush=True,
                )
    finally:
        env.close()
