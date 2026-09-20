# Departure is predictable from the input history

Finding of 20 September 2026. Evidence for P2.0 design C in
`PHASE_1_PHASE_2_RESEARCH_PLAN.md`. Reproduce with `qa/departure_predictability.py`.

**A vessel motionless throughout its twenty-minute history, which then sails within the hour, can be
identified from that history alone. AUC 0.961 for Port_Service and 0.999 for Tanker, against base
rates near 1.5%. Design C's first stage is viable. Tidal phase adds nothing.**

## Why this was measured

P2.0 design C is a two-stage model: stage 1 predicts whether the vessel moves at all, stage 2 predicts
displacement given movement. That is only worth building if stage 1 is learnable, and there was a
specific reason to doubt it. A K=2 mixture trained on Port_Service put **identical weights on windows
that stayed and windows that departed** — 0.462 against 0.464 — so an unsupervised gate extracted
nothing. The open question was whether the information was absent from the input, or merely
inaccessible to that architecture.

It is the architecture. The information is there.

## Result

Population: `not_under_way` windows with **max SOG below 0.5 kn at every input step** — vessels showing
no motion at all, which is the case design C exists for. Label: mean implied speed over the 60-minute
horizon at or above 0.5 kn. Classifier: gradient boosting on summary features of the input window.

| Group | test events | base rate | traj | +time | +tide |
|---|---:|---:|---:|---:|---:|
| Port_Service | 279 | 1.51% | **0.961** | 0.952 | 0.963 |
| Tanker | 134 | 1.49% | **0.999** | 0.999 | 0.998 |

Average precision is 0.254 for Port_Service and 0.959 for Tanker, against base rates near 0.015 — a
lift of roughly 17x and 64x.

### It is timing, not just location

The obvious objection is that the classifier has learned which berths have turnover rather than when a
particular vessel will sail. Two checks say otherwise:

- **Within-vessel AUC**, where position is nearly constant: median **0.968** across 12 Port_Service
  vessels, minimum 0.637, and **none below 0.6**. For the same vessel at the same berth, the hour
  before departure is separable from every other hour.
- **Dropping position entirely**: AUC falls only to **0.907** (Port_Service) and **0.979** (Tanker).
  Position helps and is legitimately available at prediction time, but it is not carrying the result.

Permutation importance ranks `east` and `north` far above everything else, which contradicts the
ablation. Both are correct: position is redundant with the remaining features, so permuting it hurts
while removing it lets the model substitute. Do not read permutation importance alone as evidence that
a feature is necessary.

### AIS reporting cadence is not the mechanism

Class A AIS transmits every few minutes at a berth and every few seconds under way, so a shortening
`observation_age` looked like a promising leading indicator. Measured, it is not: median `age_last` is
28 s for departing windows against 26 s for staying ones, and the age features rank near zero. The
signal is in the track, not the transmission schedule.

## Tide adds nothing, and that is a real negative result

Lock gates at Liverpool open near high water, so departures are physically tide-gated. The hypothesis
was that sea level would predict departure where the vessel's own track could not, which would have
linked the environmental layer directly to the regime work.

It does not. Adding `zos` and its rate moves Port_Service from 0.961 to 0.963 and Tanker from 0.999 to
0.998 — inside the variation between random shard draws.

The honest reading is not that tide is irrelevant to departures, but that **whatever tidal
constraint exists is already expressed in the vessel's behaviour** by the time the window ends. It adds
nothing on top. This is consistent with the environmental join's expectation of a small effect, and it
removes one specific hoped-for mechanism.

## Three measurement errors, recorded because each was silent

**1. Sampling the first N shards invalidated the first run.** Shards are written per source bucket, so
`head(40)` drew a Port_Service training set in which **every departure came from one vessel**. It
reported AUC 0.969 while measuring nothing. Sampling is now seeded choice across the split, as
`select_plan` does.

**2. Window counts overstate the evidence about thirteenfold.** Windows stride by 60 seconds, so one
departure produces 12-20 near-identical labelled windows. The full Port_Service test split has 6,103
departing windows but only about 384 independent events. **Quote events.** Any confidence interval
computed from window counts is far too narrow.

**3. `hash()` is randomised per process.** Seeding the shard draw with Python's `hash()` gave a
different sample every run, moving pooled AUC between 0.951 and 0.970 with no code change. Now seeded
with `hashlib.sha256`, matching `select_plan`.

## Caveats

- **The splits are temporal, not vessel-disjoint.** 18 of 22 Port_Service test MMSIs also appear in
  train; the `purged` and `holdout` roles confirm this is a purged time split by design. The
  within-vessel check above is what makes the result meaningful under that design, not the pooled AUC.
- **Few departing vessels.** 13 in the Port_Service test sample, and only 2 Tanker vessels met the
  within-vessel threshold. Tanker's 0.999 rests on a narrow base and should not be quoted without it.
- The 0.5 kn threshold is inherited from `motion_regime` and remains unjustified; see the open item in
  `REGIME_BIMODALITY.md`.
- Gradient boosting on summary features is a lower bound on what is learnable, not an upper one.

## What follows

1. **Build design C.** Stage 1 is viable at roughly 0.96 AUC on the hard population.
2. **Stage 1 must be calibrated, not merely accurate.** P2.0 already names the weakness: "stage 1
   miscalibration propagates invisibly". At a 1.5% base rate a classifier can be sharp and badly
   calibrated at once, and stage 2 inherits the error silently.
3. **Do not use a learned gate.** The mixture's failure was not a lack of signal. A supervised label
   reaches 0.96 on the same inputs; an unsupervised gate chasing likelihood collapses onto the
   stationary mode.
4. **Expect no help from tide here.** Whatever the environmental layer contributes, it is not
   departure timing.
