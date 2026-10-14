"""Smooth throttle by imitation: teach the driver to feed the power in.

The memoryless network can't rate-limit its own throttle (it maps what it sees
right now to a pedal position, and small changes in the rays flip it back and
forth: ~600 throttle reversals a lap on the 1:19.83 car), and random weight
search can't fix that. So this rewrites the throttle by supervised learning,
DAgger style:

1. Drive the current network round the lap (the official starts, plus a few
   runs with a little throttle noise so it sees states just off its line) and
   record every observation.
2. Label each state with a smoothed version of the *original* driver's
   throttle along that run: brake values kept as they are (same braking
   points), the throttle side low-pass filtered, trimmed where the rears
   spun, and applied progressively (at most full pedal per apply_s seconds,
   lifting is free).
3. Fit the throttle to the labels, with steering frozen: the network is
   widened by `extra` hidden units per layer that feed only the throttle, and
   the original weights that feed steering are never changed, so for any
   input the steer output is bit-identical.
4. Score the result on the official laps (lap + slides + throttle travel, the
   same Score as smooth-inputs), keep the best, and repeat on the new
   driver's states (the dataset accumulates across rounds).

Nothing here is specific to a rules version or track: sizes, rates and the
car come from the rules and the agent file. The baseline file is never
written; the best driver goes to winners/<name>-<track>.json.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch

import fevo

from .paths import winner_path
from .smooth import Score, fmt, measure


# --------------------------------------------------------------------------- network


def forward(mlp: dict[str, Any], obs: np.ndarray) -> np.ndarray:
    """(steer, throttle) for a batch of observations, as the website evaluates an MLP file."""
    h = obs
    n = len(mlp["weights"])
    for l in range(n):
        w = np.array(mlp["weights"][l]).reshape(mlp["layers"][l + 1], mlp["layers"][l])
        h = h @ w.T + np.array(mlp["biases"][l])
        if l < n - 1:
            h = np.tanh(h)
    return h


class ThrottleNet(torch.nn.Module):
    """A 2-hidden-layer tanh MLP widened by `extra` units per layer that only feed the throttle.

    Gradient masks keep every number that reaches the steer output fixed."""

    def __init__(self, mlp: dict[str, Any], extra: int, seed: int) -> None:
        super().__init__()
        L = mlp["layers"]
        assert len(L) == 4 and L[-1] == 2, "expects [inputs, h1, h2, 2]"
        n_in, h1, h2 = L[0], L[1], L[2]
        g = torch.Generator().manual_seed(seed)
        w = [torch.tensor(mlp["weights"][i], dtype=torch.float64) for i in range(3)]
        b = [torch.tensor(mlp["biases"][i], dtype=torch.float64) for i in range(3)]
        H1, H2 = h1 + extra, h2 + extra
        W0 = torch.zeros(H1, n_in, dtype=torch.float64)
        W0[:h1] = w[0].view(h1, n_in)
        W0[h1:] = torch.randn(extra, n_in, generator=g, dtype=torch.float64) / n_in**0.5
        W1 = torch.zeros(H2, H1, dtype=torch.float64)
        W1[:h2, :h1] = w[1].view(h2, h1)
        W1[h2:] = torch.randn(extra, H1, generator=g, dtype=torch.float64) / H1**0.5
        W2 = torch.zeros(2, H2, dtype=torch.float64)
        W2[:, :h2] = w[2].view(2, h2)
        B0 = torch.cat([b[0], torch.zeros(extra, dtype=torch.float64)])
        B1 = torch.cat([b[1], torch.zeros(extra, dtype=torch.float64)])
        self.W0, self.W1, self.W2 = (torch.nn.Parameter(t) for t in (W0, W1, W2))
        self.B0, self.B1, self.B2 = (torch.nn.Parameter(t) for t in (B0, B1, b[2].clone()))
        # Trainable: the new units, and the throttle row of the output layer.
        self.masks = {
            "W0": torch.zeros_like(W0), "B0": torch.zeros_like(B0),
            "W1": torch.zeros_like(W1), "B1": torch.zeros_like(B1),
            "W2": torch.zeros_like(W2), "B2": torch.zeros(2, dtype=torch.float64),
        }
        self.masks["W0"][h1:] = 1
        self.masks["B0"][h1:] = 1
        self.masks["W1"][h2:] = 1  # new units may read every layer-1 unit; old units never read new ones
        self.masks["B1"][h2:] = 1
        self.masks["W2"][1] = 1  # throttle row only; the steer row's weights to new units stay 0
        self.masks["B2"][1] = 1
        self.setup = mlp.get("setup")
        self.meta = {k: mlp[k] for k in ("activation", "outputActivation") if k in mlp}

    def throttle(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(obs @ self.W0.T + self.B0)
        h = torch.tanh(h @ self.W1.T + self.B1)
        return h @ self.W2[1] + self.B2[1]

    def mask_grads(self) -> None:
        for name, m in self.masks.items():
            p = getattr(self, name)
            if p.grad is not None:
                p.grad *= m

    def to_mlp(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "layers": [self.W0.shape[1], self.W0.shape[0], self.W1.shape[0], 2],
            "activation": self.meta.get("activation", "tanh"),
            "outputActivation": self.meta.get("outputActivation", "identity"),
            "weights": [p.detach().flatten().tolist() for p in (self.W0, self.W1, self.W2)],
            "biases": [p.detach().tolist() for p in (self.B0, self.B1, self.B2)],
        }
        if self.setup is not None:
            out["setup"] = self.setup
        return out


# --------------------------------------------------------------------------- data


def rollout(sim: fevo.Simulator, mlp: dict[str, Any], seed: int, noise: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Drive one lap from an official start (Python forward pass, optional throttle noise) and record it."""
    sim.load_program(fevo.from_json(json.dumps(mlp)))  # sets the car setup
    obs = np.array(sim.reset(seed), dtype=float)
    n_obs = len(obs)
    rows, rear, thr = [], [], []
    while True:
        steer, t = forward(mlp, obs[None])[0]
        t = float(np.clip(t, -1, 1))
        if noise:
            t = float(np.clip(t + rng.normal(0, noise), -1, 1))
        rows.append(obs)
        code, vals = sim.step_fast(float(np.clip(steer, -1, 1)), t)
        thr.append(t)
        rear.append(vals[n_obs + 7])
        obs = np.array(vals[:n_obs], dtype=float)
        if fevo.sim.STATUS[code] != "running":
            break
    return {"obs": np.array(rows), "rear": np.array(rear), "thr": np.array(thr)}


def smooth_target(raw: np.ndarray, rear: np.ndarray, hz: int, window_s: float, apply_s: float, spin_trim: float) -> np.ndarray:
    """The original driver's throttle along a run, as a smooth driver would apply it."""
    t = np.clip(raw, -1, 1)
    brake = t < 0
    w = max(1, int(round(window_s * hz)))
    tp = np.convolve(np.maximum(t, 0), np.ones(w) / w, mode="same")
    tp = np.where((rear > 0) & ~brake, tp * (1 - spin_trim), tp)
    up = 1.0 / max(1.0, apply_s * hz)
    y = np.zeros_like(tp)
    prev = 0.0
    for i in range(len(tp)):
        prev = 0.0 if brake[i] else min(tp[i], prev + up)
        y[i] = prev
    return np.where(brake, t, y)


def fit(net: ThrottleNet, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, seed: int) -> float:
    """Supervised fit of the throttle; saturated targets (±1) only ask the output to reach the limit."""
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    loss_val = 0.0
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 512):
            idx = perm[i : i + 512]
            out, y = net.throttle(Xt[idx]), Yt[idx]
            err = torch.where(y >= 0.999, torch.relu(1 - out), torch.where(y <= -0.999, torch.relu(out + 1), out - y))
            loss = (err**2).mean()
            opt.zero_grad()
            loss.backward()
            net.mask_grads()
            opt.step()
            loss_val = float(loss.detach())
    return loss_val


# --------------------------------------------------------------------------- loop


def imitate(track: str, rules: str, baseline: str | None, name: str, rounds: int, extra: int, window_s: float, apply_s: float,
            spin_trim: float, noise_runs: int, noise: float, epochs: int, lr: float, sc: Score, seed: int) -> None:
    src = baseline or str(winner_path("practice", track))
    base = json.loads(open(src).read())
    out = winner_path(name, track)
    if os.path.abspath(out) == os.path.abspath(src):
        raise SystemExit(f"{out} is the baseline; pick another --name")
    sim = fevo.Simulator(rules)
    sim.configure()
    sim.load_track(track)
    hz = sim.rules.agent_hz
    seeds = range(len(sim.rules.seeds))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    base_m = measure(base, sim)
    base_lap = base_m["lap_s"]
    best_s, best_m, best = sc(base_m, base_lap), base_m, None
    print(f"Smooth throttle by imitation at {track} ({rules}). Baseline {os.path.basename(src)}: {fmt(base_m)}", flush=True)
    print(sc.describe(base_m["lap_time"]), flush=True)
    print(f"target: throttle low-passed over {window_s} s, x{1 - spin_trim} where the rears spun, full pedal in no less than {apply_s} s; brakes kept", flush=True)

    net = ThrottleNet(base, extra, seed)
    cur = net.to_mlp()
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    t0 = time.time()
    for r in range(1, rounds + 1):
        runs = [rollout(sim, cur, s, 0.0, rng) for s in seeds]
        runs += [rollout(sim, cur, int(rng.integers(len(seeds))), noise, rng) for _ in range(noise_runs)]
        for run in runs:
            expert = forward(base, run["obs"])[:, 1]  # the original driver's throttle in these states
            X.append(run["obs"])
            Y.append(smooth_target(expert, run["rear"], hz, window_s, apply_s, spin_trim))
        loss = fit(net, np.concatenate(X), np.concatenate(Y), epochs, lr, seed + r)
        cur = net.to_mlp()
        m = measure(cur, sim)
        s = sc(m, base_lap)
        mark = ""
        if s < best_s:
            best_s, best_m, best = s, m, cur
            out.write_text(json.dumps(cur) + "\n")
            mark = "  << new best"
        n = sum(len(x) for x in X)
        print(f"round {r:2d}  {(time.time() - t0) / 60:5.1f} min  {n:6d} states  fit loss {loss:.4f}  {fmt(m)}  score {s:.3f}{mark}", flush=True)
    print(f"\nBaseline {fmt(base_m)}\nBest     {fmt(best_m)}", flush=True)
    print(f"Upload file: {out}" if best is not None else "No round beat the baseline; nothing written.", flush=True)
