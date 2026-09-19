# Observed tide gauge as an environmental source

Assessment of 19 September 2026. Companion to `ENVIRONMENTAL_JOIN_PLAN.md`, which covers the
modelled sources (Copernicus Marine currents, sea level and waves; ERA5 wind).

**Verdict: the argument this was proposed on is wrong, and measurement refutes it. A different
argument survives and is stronger. Do not act on either until the ablation has run.**

## The proposal, and why it was made

A tide gauge at Liverpool (Gladstone Dock) sits physically inside the dock estate where the
near-berth groups operate. The 1.5 km ocean model has no water there — Step 0 measured only 11.1%
of Tanker window origins and 14.7% of Passenger origins landing on a wet **current** cell. The
proposal was that an instrument inside the estate would serve the windows the model cannot.

## Refuted by measurement

Measured on the test split of `full_5_60_20260918T072424Z`, up to six shards per group, sampling
`zos` from `env_cache/ssh_1p5km.nc` through the same `Field` class the sidecar builder uses:

| Group | Windows | `zos` resolved | Median sampled cell distance |
|---|---:|---:|---:|
| Cargo | 11,366 | **100.0%** | 0.0 km |
| Unknown | 27,643 | 99.8% | 0.0 km |
| Tanker | 10,912 | **93.4%** | 0.0 km |
| Port_Service | 30,000 | **100.0%** | 0.0 km |
| Passenger | 14,241 | **100.0%** | 0.0 km |
| Fishing | 1,232 | 91.2% | 0.0 km |
| Research_Offshore | 24,500 | 100.0% | 0.0 km |

**Sea level is already available for 93–100% of windows in every group, at a median sampling
distance of zero.** The coverage failure the gauge was proposed to fix does not exist for `zos`.

Two things were conflated in the proposal, and the plan had already distinguished them:

1. The 11.1% / 14.7% figures are **currents** (`uo`/`vo`, 631 wet cells), not sea level (`zos`,
   670 wet cells). The plan's own section "Consequence — `zos` may be the better tidal feature"
   predicted exactly this and was correct.
2. Those figures are **exact-cell** lookup. The builder does not use exact-cell lookup; the
   nearest-wet-cell decision recorded in Step 0 already closed the gap.

The residual misses do not help the case either. Of the windows where `zos` cannot be resolved,
**0.0% lie within 10 km of the gauge** in every group — Tanker's 723 misses included. The windows
the model cannot serve are precisely the ones a Gladstone gauge also cannot serve; they are far
up-river or outside the downloaded box, not in the dock estate.

This is the second time in this project that a claim about near-berth coverage has survived until
it was measured and then failed. The pattern is worth noting: the coverage intuition is
consistently pessimistic relative to what nearest-wet-cell sampling actually delivers.

## What survives: the pre-August-2024 training gap

The 1.5 km marine archive begins 2024-08-04. The AIS release begins 2023-08-22. Measured exactly
over all 4,912 shards:

| Split | Windows | Covered at 1.5 km |
|---|---:|---:|
| train | 16,096,739 | **56.3%** |
| validation | 1,393,326 | 100.0% |
| calibration | 662,668 | 100.0% |
| test | 1,949,571 | 100.0% |

**7,041,564 training windows — 43.7% of training, 35.0% of the release — predate the high-resolution
archive.** The plan's recommendation is to discard them rather than mix 7 km and 1.5 km forcing,
because mixed-resolution forcing introduces a covariate shift indistinguishable from the
environmental effect being measured.

A tide gauge is the one source that does not face this trade. It is **a single instrument at a
single location across the entire AIS span** — no resolution change, no product switch, no
covariate shift between the early and late training periods. It would give a tidal-phase feature
for 100% of windows where the model gives one for 56.3%.

That is a materially better argument than the one the gauge was proposed on, and it is orthogonal
to what the ocean model provides rather than duplicating it.

## Representativeness: good for the estuary, useless offshore

A gauge is a point. It carries no spatial variation, so it is only usable where the fleet is
close enough for one number to be meaningful. Distance from window origin to the gauge
(53.4497 N, 3.0184 W):

| Group | Median distance | Within 5 km | Within 10 km | Within 20 km |
|---|---:|---:|---:|---:|
| Cargo | 1.5 km | 65.1% | 74.4% | 87.5% |
| Unknown | 2.4 km | 85.7% | 90.6% | 98.4% |
| Tanker | 2.5 km | 54.7% | 63.8% | 91.5% |
| Port_Service | 2.6 km | 88.6% | **97.9%** | 99.9% |
| Passenger | 5.2 km | 37.5% | 64.9% | 91.1% |
| Fishing | 13.6 km | 23.1% | 36.4% | 82.7% |
| Research_Offshore | **37.5 km** | 0.1% | **0.2%** | 0.6% |

The estuary groups cluster tightly around the gauge — Port_Service has 97.9% of windows within
10 km. **Research_Offshore is disqualified outright** at a median of 37.5 km, and Fishing is
marginal. Any gauge feature must therefore be per-group or distance-gated, not applied fleet-wide.

Note the ordering: the groups closest to the gauge are exactly the groups the model already serves
at 100%. The gauge's representativeness and its necessity are inversely related — which is the
core reason this is a weak option in the present, and only becomes interesting for the historical
period.

## Where a gauge genuinely beats the alternative

For the pre-2024-08 period the alternative is not the 1.5 km product — it does not exist yet. It is
the **7 km reanalysis**, which the plan measured as having only 32 wet cells in the footprint box,
with the Mersey channel not resolved at all.

For the estuary groups, an instrument in the dock plausibly beats a 7 km cell that does not resolve
the river. For Research_Offshore, the 7 km cell is the better source. That is the real shape of the
decision, and it is narrower than "use observations instead of models".

## If it were built

Mirror the existing `ssh` block rather than inventing a parallel design:

- `gauge_level` and `gauge_rate` — level and its time derivative, matching the existing
  `ssh` / `ssh_rate` pair. Sea level and tidal stream run roughly in quadrature in an estuary, so
  level plus rate places a window in the tidal cycle without needing a vector.
- `gauge_distance_km` — recorded per window, as `*_cell_distance_m` already is, so results can be
  stratified by it and the same falsification test applies: a real effect should decay with distance
  from the instrument.
- A separate validity flag, per the plan's rule against shared masks.

**Datum warning.** Model `zos` is referenced to the geoid; a gauge is referenced to chart datum
(Admiralty Chart Datum at Liverpool). The two are offset by a constant of several metres. They must
not share a column or be treated as interchangeable — carry the gauge as its own feature, or
standardise each separately before they meet.

**Access.** The UK National Tide Gauge Network is held by BODC, not CEDA, so this is a separate
route from the JASMIN/CEDA path and does not benefit from JASMIN co-location. Availability,
cadence and the current download mechanism need confirming before any of this is built.

## Recommendation

Do not build this now. In order:

1. Finish the stability rerun and the regime separation, as already sequenced.
2. Run ablations A–C on the 1.5 km era alone. If environmental data moves nothing, this question
   is closed and the gauge is irrelevant.
3. **Only if a tidal effect is demonstrated**, revisit the gauge — as a way to extend that effect
   to the 43.7% of training windows the high-resolution archive cannot reach, for the estuary
   groups only.

Step 3 is the sole surviving justification. The coverage argument is withdrawn.
