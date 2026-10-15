"""Teach a fresh network to drive like the racing-line expert (fevoracer.racer).

DAgger: in round 0 the expert drives and the network learns its (steer,
throttle) from the 15 official inputs; in later rounds the network drives
(the expert takes over on a shrinking share of steps), the expert labels
every state the network gets itself into, and the network trains on
everything so far. Each round drives the three official starts and a batch
of practice starts anywhere on the lap at around the expert's speed.

The network is a plain 15-input MLP (tanh hidden layers, linear output) for
the car setup given; a round's network is kept only if it finishes all
three official starts, and the fastest goes to winners/<name>-<track>.json.

    uv run python -m fevoracer.dagger --rules solo-2 --setup-from winners/solo2safe-spielberg.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn

import fevo

from .paths import CHECKPOINTS, winner_path
from .racer import LineExpert

MAX_FILE = 500 * 1024


class Student(nn.Module):
    def __init__(self, hidden: list[int], n_in: int = 15) -> None:
        super().__init__()
        dims = [n_in, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def from_mlp(cls, mlp: dict) -> "Student":
        """A student holding an existing tanh MLP's weights (a warm start)."""
        assert mlp["activation"] == "tanh" and mlp["outputActivation"] == "identity" and mlp["layers"][-1] == 2
        st = cls(mlp["layers"][1:-1], mlp["layers"][0])
        lin = [m for m in st.net if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for m, w, b in zip(lin, mlp["weights"], mlp["biases"]):
                m.weight.copy_(torch.tensor(w, dtype=torch.float32).view(m.out_features, m.in_features))
                m.bias.copy_(torch.tensor(b, dtype=torch.float32))
        return st

    def to_mlp(self, setup: dict[str, float] | None, digits: int = 6) -> dict:
        lin = [m for m in self.net if isinstance(m, nn.Linear)]
        r = lambda t: [float(f"{v:.{digits}g}") for v in t.detach().double().flatten().tolist()]  # noqa: E731
        out = {"layers": [lin[0].in_features] + [m.out_features for m in lin], "activation": "tanh", "outputActivation": "identity",
               "weights": [r(m.weight) for m in lin], "biases": [r(m.bias) for m in lin]}
        if setup:
            out["setup"] = setup
        return out


def rollout(sim: fevo.Simulator, expert: LineExpert, student: Student | None, start: dict, max_steps: int, beta: float,
            rng: random.Random) -> tuple[list, list, str, int]:
    """One episode; each step the expert drives with probability beta, else the student. Returns states, expert labels."""
    if "seed" in start:
        obs = sim.reset(start["seed"])
    else:
        obs = sim.reset(1, point=start["point"], lateral_m=start["lat"], heading_rad=start["hdg"], speed=start["speed"])
    expert.reset()
    xs, ys = [], []
    code = 0
    for _ in range(max_steps):
        a_exp = expert(sim, obs)
        xs.append(list(obs))
        ys.append(a_exp)
        if student is not None and rng.random() >= beta:
            with torch.no_grad():
                a = student(torch.tensor([obs], dtype=torch.float32))[0].clamp(-1, 1).tolist()
        else:
            a = a_exp
        code, vals = sim.step_fast(a[0], a[1])
        obs = list(vals[:15])
        if code != 0:
            break
    return xs, ys, fevo.sim.STATUS[code], sim.progress_index


def fit(student: Student, X: np.ndarray, Y: np.ndarray, steps: int, lr: float, steer_weight: float, batch: int = 1024,
        sample_weight: np.ndarray | None = None, drop_cols: tuple[int, ...] = (), drop_p: float = 0.0) -> float:
    """`steps` minibatch updates (Adam, learning rate decayed tenfold by the end); returns the full-data loss.

    sample_weight draws states in proportion to it (e.g. more braking zones) instead of uniformly. drop_cols are zeroed
    together in a drop_p share of each minibatch's states (dropout on inputs the network should not lean on alone)."""
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 10)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    w = torch.tensor([steer_weight, 1.0])
    sw = None if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32)
    for _ in range(steps):
        i = torch.randint(len(Xt), (batch,)) if sw is None else torch.multinomial(sw, batch, replacement=True)
        xb = Xt[i]
        if drop_cols and drop_p > 0:
            xb = xb.clone()
            gone = torch.rand(batch) < drop_p
            xb[gone.nonzero().unsqueeze(1), torch.tensor(drop_cols)] = 0.0
        loss = ((student(xb) - Yt[i]).pow(2) * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return float(((student(Xt) - Yt).pow(2) * w).mean())


def official(mlp: dict, track: str, rules: str) -> fevo.OfficialResult:
    return fevo.evaluate(fevo.from_json(json.dumps(mlp)), track, rules)


def dagger(track_id: str, rules: str, setup_from: str | None, name: str, rounds: int, hidden: list[int], practice: int,
           steps: int, lr: float, steer_weight: float, seed: int, expert_kw: dict, init: str | None = None) -> None:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    sim = fevo.Simulator(rules)
    track = sim.load_track(track_id)
    setup_d = json.load(open(setup_from))["setup"] if setup_from else None
    setup = [setup_d[k["id"]] for k in sim.rules.setup] if setup_d else None
    sim.set_setup(setup)
    expert = LineExpert(track, sim.rules, setup, **expert_kw)
    student = Student.from_mlp(json.load(open(init))) if init else Student(hidden)
    if init:
        hidden = [m.out_features for m in student.net if isinstance(m, nn.Linear)][:-1]
        warm = official(student.to_mlp(setup_d), track_id, rules)
        print(f"warm start from {init}: official {' / '.join(fevo.format_lap_time(l.steps) if l.finished else l.status for l in warm.laps)}", flush=True)
    out = winner_path(name, track_id)
    X: list[list[float]] = []
    Y: list[tuple[float, float]] = []
    best = None
    t0 = time.time()
    print(f"DAgger at {track_id} ({rules}), student {[15, *hidden, 2]}, setup {setup_d}, expert {expert_kw}", flush=True)
    for r in range(rounds + 1):
        beta = 1.0 if r == 0 else max(0.0, 0.6 - 0.1 * r)
        starts: list[dict] = [{"seed": s % 3} for s in range(6)]
        for _ in range(practice):
            p = rng.randrange(track.points)
            starts.append({"point": p, "lat": rng.uniform(-2.0, 2.0), "hdg": rng.uniform(-0.06, 0.06),
                           "speed": rng.uniform(0.7, 1.0) * float(expert.v[p])})
        ends = []
        for st in starts:
            xs, ys, status, idx = rollout(sim, expert, None if r == 0 else student, st, 6000 if "seed" in st else 700, beta, rng)
            X += xs
            Y += ys
            if "seed" in st:
                ends.append(status if status == "finished" else f"{status}@{idx}")
        loss = fit(student, np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), steps, lr, steer_weight)
        mlp = student.to_mlp(setup_d)
        res = official(mlp, track_id, rules)
        clean = all(l.finished for l in res.laps)
        laps = " / ".join(fevo.format_lap_time(l.steps) if l.finished else f"{l.status}@gate {l.gates_passed}" for l in res.laps)
        mark = ""
        if clean and (best is None or res.median_steps < best):
            best = res.median_steps
            out.write_text(json.dumps(mlp) + "\n")
            mark = "  << new best"
        size = len(json.dumps(mlp))
        print(f"round {r:2d}  beta {beta:.1f}  {len(X):8,d} states  loss {loss:.5f}  rollouts from seeds {ends}  "
              f"official {laps}  ({size / 1024:.0f} KB, {(time.time() - t0) / 60:.1f} min){mark}", flush=True)
    np.savez_compressed(CHECKPOINTS / track_id / f"{name}-demos.npz", X=np.array(X, dtype=np.float32), Y=np.array(Y, dtype=np.float32))
    torch.save(student.state_dict(), CHECKPOINTS / track_id / f"{name}-student.pt")
    print(f"\nBest clean official lap: {fevo.format_lap_time(best) if best else 'none'}  ->  {out if best else 'nothing written'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="dagger", description="Imitate the racing-line expert into a 15-input MLP (DAgger)")
    ap.add_argument("--track", default="spielberg", choices=fevo.track_ids())
    ap.add_argument("--rules", default="solo-2")
    ap.add_argument("--setup-from", help="agent file whose setup the car (and the written network) uses")
    ap.add_argument("--name", default="dagger", help="writes winners/<name>-<track>.json")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--practice", type=int, default=120, help="practice starts per round")
    ap.add_argument("--steps", type=int, default=20000, help="minibatch updates per round")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steer-weight", type=float, default=4.0)
    ap.add_argument("--expert", default="{}", help="LineExpert keyword arguments as JSON")
    ap.add_argument("--init", help="agent file (a 15-input tanh MLP) to warm-start the student from")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if os.path.abspath(winner_path(a.name, a.track)) in {os.path.abspath(p) for p in [a.setup_from or ""]}:
        raise SystemExit("--name would overwrite --setup-from")
    dagger(a.track, a.rules, a.setup_from, a.name, a.rounds, [int(h) for h in a.hidden.split(",")], a.practice, a.steps, a.lr,
           a.steer_weight, a.seed, json.loads(a.expert), a.init)


if __name__ == "__main__":
    main()
