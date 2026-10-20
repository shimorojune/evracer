# Training handoff (updated 2026-10-16)

Driver training is **frozen**. Rohit chose "Build the site" as fevo's main goal (2026-10-16); the driver below is the
house entry. Nothing is running. Don't start new training directions without Rohit asking.

## House driver (frozen)

`winners/house-driver-solo6.json` (= `general-solo6-pace2`), rules **solo-6** (39 inputs, simulator 1.5.0).

- Held-out (13 unseen tracks x 5 laps): **55/65 clean**, 45 at 3% action error, 37 at 6%, 12 at 10%
  (`FEVO_RULES=solo-6 python -m fevoracer.robust_check`, drifting steering/pedal error, 3 seeds).
- Pace vs expert where it laps: +3-6% on generated tracks, Spielberg +6%, Spa +8%, Montreal +9%, Austin +12%.
- Fails: Monza (first chicane: does not move to the far edge before turn-in, misses the second apex) and g900006.
- Comparison: solo-5 best general-solo4-es 26/15/4/1.
- Practice from the predecessor r4 on the 7 test tracks: all 3/3 within 24-123 laps (`winners/practice-solo6-r4/`).

## How it was made (solo-6, 2026-10-15/16)

1. Widened general-solo4-es to 39 inputs, zero weights: bit-identical on solo-6 (`widen_check.py`).
2. DAgger with action noise 0.05 and prev_* dropout 0.3: `general-solo6` 30/65.
3. ES: `general-solo6-es` 45/65 but brittle (24 at 3% error).
4. Robustness ES (`general_es --lap-noise --robust-select`), then `--mix-tracks 3` (late chicanes, fast kinks; fixed
   Spa's top-speed drift and Montreal's last chicane), then errors up to 10%: `general-solo6-r4` 54/48/36/13.
5. Pace: `--lap-both --act-noise 0.06 --robust-select 0.06 --pace-select` (lap driven clean and with error; fastest
   net above the start's robustness floor): `general-solo6-pace2`. Lighter noise alone was fast but brittle.
6. Monza attempts, both failed: `chic:` generated circuits (tight 15-28 m chicane at the end of the longest straight,
   `trackgen.generate(chic=True)`; the house driver fails most of them too). ES 80 gens on them: Monza unchanged,
   nothing above the floor. DAgger 8 rounds (`--chic-share 0.4`): collapsed to 5-15 clean laps, Monza unchanged
   (one early averaged net got 48% round Monza). Logs: `general-solo6-chic.log`, `general-solo6-chicd.log`.

ES and DAgger now save every scored network in `checkpoints/` (`<name>-genNN.json`, `<name>-rN-new|avg.json`).

## Earlier results (solo-5; training still defaults to solo-5 via `general.RULES` / `FEVO_RULES`, use FEVO_RULES=solo-6)

- Best general driver: `general-solo4-es`, 26/65 held-out laps, 59.7% (solo-4: 27, 60.7%).
- Practice, 7 unseen test tracks, all 3/3 after 2000 laps: Spielberg 1:12.52 (+1.9% vs expert), g900001 1:02.85,
  g900006 1:00.67, Monza 1:34.40, Montreal 1:26.95, Spa 1:58.80, Austin 1:58.23 (+13.6%). `winners/practice-solo5/`.
- Notebooks (v2 rules) on solo-5: modest gains over 30 laps, about 30% of practice laps crash. `winners/notebook-solo5/`.
- Distillation on solo-5 (39 teachers): no round beat the starting driver (best 23 laps). Teachers removed in the repo cleanup (in git history before it).
- Speed: agent evaluation 4.1x faster (173 -> 42 us/step, fevo `9bf0702`), bit-identical; a 2000-lap practice
  run 34.3 -> 5.6 min with byte-identical checkpoints. Distillation practises hardest tracks first.

## Commits (2026-10-15; solo-6 training commits since are in `git log`)

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
  `distill.py`, `robust_check.py`, `widen_check.py`, `practice.py` / `practice2.py` (practice: survival, focused survival, pace), `rescore.py`.
- Notebook: `notebook2.py` (current rules), `notebook_lab.py` + `notebook_lab.html` -> `notebook-lab.html`,
  `notebook_replay.py` (replays exported laps exactly), `notebook_drag_check.py`.
- Checks: `hashes.py` (bit-identical proof of older rules versions), `loophole_scan.py`, `retime.py --wasm`,
  `scrub_check.py`, `slide_check.py`, `expert_check.py`.
- Logs, proposals, `notebook-lab.html` and older practice/distillation outputs were removed in the repo cleanup; recover them from git history (`git show c4e0333:<path>`).

## Open items (parked)

- Push the 15 fevo commits (solo-2 to solo-6, replay viewer, evaluator speed-up): Rohit's to push.
- If training resumes: Monza-type chicanes (maybe a bigger network or line-aware inputs), solo-6 in the notebook lab,
  pace closer to the expert.

## Conventions

- Every simulator change proves all older rules versions bit-identical (`python -m fevoracer.hashes` before/after,
  plus the fevo parity fixture's earlier cases unchanged). Never edit a published rules version; add a new one.
- Rules and physics changes need Rohit's OK first. Held-out tracks (g900000-g900008, Austin, Montreal, Monza, Spa,
  Spielberg; g950000+/g960000+ for notebooks) never go into training.
- One heavy job at a time; long runs under `nohup caffeinate -s ...`, and keep the laptop lid open.
- Downloads need Rohit's OK. Commit as `git -c user.name=shimorojune -c user.email=rohitrajnovember@gmail.com`;
  never push (fevo pushes are Rohit's). fevo-racer1 has no remote.
