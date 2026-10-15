# Proposal: rules version solo-6, a short memory for the driver

Status: draft for Rohit's approval. Nothing built.

## What changes

solo-6 = solo-5 (same car, physics, track limits, timing, seeds) with six more inputs after solo-5's 33, so agents
take 39. The simulator computes them; the agent format and network type do not change (still a plain MLP).

| # | name | meaning | scale |
|---|------|---------|-------|
| 33 | `prev_steer` | the steering the car actually got last step (clamped, -1..1) | as is |
| 34 | `prev_throttle` | the pedal the car actually got last step (clamped, -1..1) | as is |
| 35 | `d_speed` | change in forward speed over the last 0.1 s (6 steps) | / 5 m/s |
| 36 | `d_yaw_rate` | change in yaw rate over the last 0.1 s | / 1 rad/s |
| 37 | `d_offset` | change in lateral offset from the centre line over the last 0.1 s | / 1 m |
| 38 | `d_heading` | change in heading relative to the track over the last 0.1 s | / 0.2 rad |

At a reset (lap start or practice stint) the previous action is 0 and the changes are 0, so a lap is still fully
determined by the rules, the track, the start and the agent: bit-identical everywhere as now. The simulator keeps a
6-step ring buffer; that is new state, reset with the lap.

It is deliberately short memory: one step of action and 0.1 s of trend. It says "I am braking", "I am sliding wide
and it is getting worse", "the rear is stepping out", not where on the track the car is, so it cannot be used to
memorise a circuit (lap progress inputs stay masked for general drivers, as now).

## Why

- The braking-onset problem: near the braking point, almost identical states need opposite actions depending on
  whether the driver has already started braking. The expert has a braking latch the network cannot see; with
  `prev_throttle` it can (hold the brake once on), which removes the main source of inconsistent labels we found.
- Brittleness: today the driver reacts only to the present, so a small error is invisible until it is large. The
  trend inputs show a slide or a drift building up 0.1 s earlier, and give it a way to damp its own oscillations.
- Smoother driving: with its last action as an input the network can learn smooth pedal and steering changes
  instead of jumping between values every step.

## How training changes

- Warm start: today's 33-input drivers are widened to 39 with zero weights on the new inputs, so they start out
  driving exactly as now on solo-6 (proved on the held-out set before anything trains).
- Imitation (DAgger): the new inputs come from whoever is driving; that is fine (DAgger is on-policy). The known risk
  is "copycat" learning: the network leans on `prev_*` and just repeats its last action. Countermeasures: noise on
  the applied action during data collection (already in practice/ES), dropout on the two `prev_*` inputs while
  fitting, and checking the held-out lap count, not the training loss.
- ES, practice and notebooks: unchanged code; they see the extra inputs automatically.
- Expert: unchanged (it reads the simulator state directly).

## How we would judge it

Same tests as before, on solo-6 vs solo-5: held-out laps finished and completion (13 circuits x 5 laps), the practice
curve on the 7 test tracks (laps to 3/3 and gap to the expert at 20/100/500/2000), and the crash rate during
practice. Success: clearly more held-out laps or a faster practice curve; if not, solo-6 is shelved.

## What it costs

- fevo: sim-core (ring buffer, six inputs, observation set code 2), rules solo-6, SDK and replay viewer support,
  tests; simulator 1.5.0. solo-1 to solo-5 bit-identical, proved as before.
- Agent files grow by about 1.2k weights (well under the 500 KB limit).
- fevo-racer1: widen the drivers, retrain the general driver (about 2 h with the faster simulator), rerun the tests.
