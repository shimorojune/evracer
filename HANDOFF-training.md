# Training handoff (2026-10-15)

Where the driver-training work stands, for the next thread. Nothing is running.

## solo-6 (short memory): built and proven, not trained

- fevo commit `e7948c1` (simulator 1.5.0): rules solo-6 = solo-5 + 6 inputs (39 in all): `prev_steer`,
  `prev_throttle` (last step's applied action), `d_speed`, `d_yaw_rate`, `d_offset`, `d_heading` (change over the last
  0.1 s). All zero after every reset. Design: `proposals/solo6-memory.md`. Logged in fevo `docs/decisions.md`.
- Proof: all fevo tests and typechecks pass (new tests: memory inputs zero after a reset, last applied action,
  0.1 s change, extended part unchanged; solo-6 rules test). solo-1 to solo-5: all 306 winner laps bit-identical
  (`logs/hashes-post-agentfast.json` vs `logs/hashes-post-solo6.json`). Parity fixture: 39 earlier cases unchanged,
  6 solo-6 cases added. SDK reports 1.5.0 with rules solo-1 to solo-6; `fevo.from_json` infers 39-input MLPs.
- Not done yet: any training, evaluation, or notebook-lab support for solo-6.

## Next steps (Rohit approved "Build solo-6" at 19:18 UTC)

1. Widen the best general driver (`winners/general-solo4-es.json`, 33 inputs) to 39 with zero weights for the six new
   inputs (first-layer rows are [out][in]; see `explore.widen`). Verify it drives identically on solo-6 to solo-5
   (same lap hashes on the held-out set) before training.
2. Retrain on solo-6 (`FEVO_RULES=solo-6`): DAgger (`general.py`) with action noise during data collection and dropout
   on the two `prev_*` inputs while fitting (guards against copycat learning), then ES (`general_es.py`).
   `general.MASKED` (lap progress inputs 12 and 32) must stay masked; check the indices by name for 39 inputs.
3. Evaluate on the same 13 held-out tracks x 5 laps (`python -m fevoracer.rescore`, laps finished first) against
   solo-5's best: general-solo4-es, 26/65 laps, 59.7%. Practice-curve comparison on a few test tracks
   (`practice2 --unlock 2 --focus-pairs 16 --stints 4 --stint 600`, compare with `winners/practice-solo5/`).
4. If there is no gain, shelve solo-6 and say so plainly.
5. Add solo-6 to `notebook-lab.html` (`notebook_lab.SWITCH`, page boot sets the observation set from rules; needs
   39-input drivers, so only once widened drivers exist; check JS vs Python parity on every track).

## Results so far (solo-5, the loophole-free physics; training defaults to solo-5 via `general.RULES` / `FEVO_RULES`)

- Best general driver: `general-solo4-es`, 26/65 held-out laps, 59.7% (solo-4: 27, 60.7%).
- Practice, 7 unseen test tracks, all 3/3 after 2000 laps: Spielberg 1:12.52 (+1.9% vs expert), g900001 1:02.85,
  g900006 1:00.67, Monza 1:34.40, Montreal 1:26.95, Spa 1:58.80, Austin 1:58.23 (+13.6%). `winners/practice-solo5/`.
- Notebooks (v2 rules) on solo-5: modest gains over 30 laps, about 30% of practice laps crash. `winners/notebook-solo5/`.
- Distillation on solo-5 (39 teachers): no round beat the starting driver (best 23 laps). `winners/distill-teachers-solo5/`.
- Speed: agent evaluation 4.1x faster (173 -> 42 us/step, fevo `9bf0702`), bit-identical; a 2000-lap practice
  run 34.3 -> 5.6 min with byte-identical checkpoints. Distillation practises hardest tracks first.

## Commits today

fevo (local, NOT pushed; all await Rohit's push): `e7948c1` solo-6, `9bf0702` evaluator speed-up, `f0b5666` solo-5,
plus earlier unpushed ones: `bbfc122`, `4dee9f4`, `e164193`, `63b352b`, `2b3f3a1`, `fc500e1`, `8ab1ff1`, `70be396`,
`21efdbc`, `ed257cd`, `aed2633`, `f9dcfa6` (15 in all ahead of origin/main).

fevo-racer1 (local only, no remote): `acbf0a3` hardest-first + speed check + solo-6 proposal, `c2706db` solo-5
distillation, `3f87c6e` solo-5 notebooks, `308fb3e` solo-5 practice, `743ce30` solo-5 default + rescore + lab switch,
`c4e9a51` hashes proof, `2f6f274` loophole scan + re-time, `0b5325b` lap export/replay, `ba4e660` car sprite,
`db2259b` distillation (solo-4), `e117ff1` slide_check, `0bc177a` scrub_check, `694c4c4` notebook learning fix,
`3f49064` distill, `d249f8e` racing-line limit (off by default), `4d0b3eb` notebook lab, `16ae1e4` notebook v2,
`a23d7c8` notebook v1, `289b090` practice4, `0709037` practice3, `499d12f` practice2, `b5e83a1` practice, `d491821`
laps-first winners, earlier ES/generator/expert commits (`eea4f40` ... `e7f1fee`). This note's commit follows.

## Key files (fevo-racer1)

- Training: `src/fevoracer/general.py` (DAgger, `RULES`, held-out set), `general_es.py` (ES, noise curriculum),
  `distill.py`, `practice.py` / `practice2.py` (practice: survival, focused survival, pace), `rescore.py`.
- Notebook: `notebook2.py` (current rules), `notebook_lab.py` + `notebook_lab.html` -> `notebook-lab.html`,
  `notebook_replay.py` (replays exported laps exactly), `notebook_drag_check.py`.
- Checks: `hashes.py` (bit-identical proof of older rules versions), `loophole_scan.py`, `retime.py --wasm`,
  `scrub_check.py`, `slide_check.py`, `expert_check.py`.
- Proposals: `proposals/solo6-memory.md`, `proposals/solo5-blend-fix.diff`.
- Logs: `logs/` (distill*.log, practice-solo5.log, notebook-solo5.log, loophole_scan.log, retime-fix.log, hashes-*).

## Open questions for Rohit

- Push the 15 fevo commits (solo-2 to solo-6, replay viewer, evaluator speed-up)?
- After solo-6: whether to try better distillation (teacher weighting, student noise, bigger student) or a
  training-only GPU sim (exact-sim re-check of every result).
- Notebook practice still crashes about 1 lap in 4; whether to make notes trainable inside the network (a later
  rules item).

## Conventions

- Every simulator change proves all older rules versions bit-identical (`python -m fevoracer.hashes` before/after,
  plus the fevo parity fixture's earlier cases unchanged). Never edit a published rules version; add a new one.
- Rules and physics changes need Rohit's OK first. Held-out tracks (g900000-g900008, Austin, Montreal, Monza, Spa,
  Spielberg; g950000+/g960000+ for notebooks) never go into training.
- One heavy job at a time; long runs under `nohup caffeinate -s ...`, and keep the laptop lid open.
- Downloads need Rohit's OK. Commit as `git -c user.name=shimorojune -c user.email=rohitrajnovember@gmail.com`;
  never push (fevo pushes are Rohit's). fevo-racer1 has no remote.
