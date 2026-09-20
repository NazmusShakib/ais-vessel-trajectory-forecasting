# Why Port_Service and Research_Offshore have no skill

Finding of 19 September 2026. Companion to `ENVIRONMENTAL_JOIN_PLAN.md`.
Reproduce with `qa/regime_structure.py` and `qa/regime_bimodality.py`.

**The two groups the model cannot beat a constant-position baseline on are not harder. Their windows
are disproportionately regime *transitions*, and the model emits a unimodal Gaussian where the true
predictive distribution has two modes. It therefore predicts the average of "stays alongside" and
"departs at ten knots", which is worse than committing to either.**

This supersedes two earlier explanations, both withdrawn on measurement.

## How this explanation was arrived at

Worth recording, because the first two answers were wrong and the third was found only by continuing
to measure:

1. **Tidal forcing.** Withdrawn 17 September. Port_Service M2 amplitude is 121 m against 2,063 m RMS
   displacement (5.8%), Cargo gives 30.4% — the opposite of the predicted ordering. Not-under-way
   Port_Service windows have a median displacement of **one metre over sixty minutes**. Moored vessels
   do not follow the tide.
2. **Regime mixture.** Partly right, and the basis for reporting error by regime. But a mixture of an
   easy population and a hard one predicts that separating them helps. Measurement below shows it
   would not, because the mixture is *within* the input regime, not across it.
3. **Distributional mismatch.** What the evidence actually supports.

## Measurement 1 — transitions carry a third of the error, in exactly the failing groups

Release 0918, test split, 8 shards per group. Model-free: the constant-position baseline's error is
exactly `||y||` averaged over the twelve horizons, which is the quantity the published skill numbers
are relative to. Horizon behaviour is "moves" when mean implied speed over 60 minutes is at least
0.5 kn.

Share of each group's total error, by input regime and what actually happened:

| Group | Skill | moored→**moves** | under way→**stops** | **transitions** |
|---|---:|---:|---:|---:|
| **Port_Service** | 1.04 | 16.0% | 17.8% | **33.8%** |
| **Research_Offshore** | 0.98 | 1.3% | 34.5% | **35.8%** |
| Unknown | 0.78 | 12.7% | 19.9% | 32.6% |
| Passenger | 0.74 | 10.7% | 21.0% | 31.7% |
| Cargo | 0.29 | 4.3% | 1.7% | **6.0%** |

The ordering is the point. The two groups with no skill are the two with ~34% of error in transition
windows; the group with the best skill has 6%.

`sog_unknown` does not occur at all in the sample, so the third regime label needs no routing
decision.

## Measurement 2 — the mixture is *inside* the input regime

60-minute net displacement within the not-under-way population alone:

| Group | p50 | p90 | p99 | % that move | error carried by those movers |
|---|---:|---:|---:|---:|---:|
| Port_Service | **0.7 m** | 11 m | **7,764 m** | 2.5% | **76.9%** |
| Tanker | 2.6 m | 11 m | **12,500 m** | 2.6% | 92.5% |
| Passenger | 0.0 m | 21,404 m | 27,647 m | 24.8% | 98.6% |
| Cargo | 11.6 m | 12,927 m | 20,187 m | 19.0% | 94.5% |
| Research_Offshore | 12.9 m | 444 m | 2,291 m | 0.6% | 17.0% |

The median Port_Service "moored" window moves **70 centimetres**; its 99th percentile moves **7.8 km**.
Research_Offshore is the one genuinely unimodal case.

**This is what rules out separate models per regime.** The two populations share an input history and
diverge only in the future, so a model selected on input regime would route departures to the
specialist trained to predict stillness.

## Measurement 3 — the trained model sits between the modes

Release 0913 test split, 6 shards per group, the `bilstm_attention` runs of 15 September. Those models
were trained against 0913 — verified by manifest SHA-256 `71d02d63…` and by Cargo's test count of
47,225 — so they are evaluated on 0913's own test split. Evaluating them on 0918 would pull in windows
that were calibration or validation for these models.

Median 60-minute displacement, true against predicted:

| Group | Input → horizon | % windows | True | **Predicted** |
|---|---|---:|---:|---:|
| Port_Service | not under way → **stays** | 76.2% | **1.3 m** | **82.6 m** |
| Port_Service | not under way → **moves** | 2.3% | **6,946 m** | **301.9 m** |
| Port_Service | under way → stops | 14.6% | 1,044 m | 312.4 m |
| Port_Service | under way → moves | 6.9% | 7,015 m | 2,243.7 m |
| Research_Offshore | not under way → stays | 46.4% | 11.8 m | 381.5 m |
| Research_Offshore | not under way → moves | 2.2% | 6,695.7 m | 455.5 m |

From one input regime the model over-predicts motion **60-fold** for vessels that do not move, and
under-predicts **23-fold** for vessels that depart. Both errors at once, in opposite directions, from
the same conditional distribution. An under-trained or under-capacity model is wrong in one direction;
this is the signature of a single Gaussian straddling two modes.

## Measurement 4 — calibration is inverted across the modes

Mean error divided by median predicted sigma, within the not-under-way regime:

| Group | → stays | → moves |
|---|---:|---:|
| Port_Service | **0.4** | **1.7** |
| Research_Offshore | **0.2** | **2.0** |
| Cargo | 0.2 | 1.3 |

One sigma for a two-mode population is necessarily too wide for one and too narrow for the other.
Port_Service stationary windows carry a ~310 m uncertainty region for vessels that move 1.3 m, which
is also the explanation for the region-area figures.

## Cargo is the control, and it confirms the mechanism

Cargo has the best skill of any group (0.29). **65.6% of its windows are under way → moves, carrying
83% of its error, and there the model predicts 18,781 m against a true 19,432 m** — within 3.4%.

Same architecture, same loss, same features: accurate where the target distribution is unimodal,
incoherent where it is not. The failure is not the model's capacity, it is the shape of its output.

## What follows

**Ruled out.** Separate models per input regime — Measurement 2 shows the split is inside the regime.
A regime *flag* as an input feature is close to a no-op in any case: `motion_regime` is derived from
SOG, which is already input feature index 2 across all twenty steps, so the model has the information
and the problem is not that it cannot see the regime.

**Indicated.** A predictive distribution that can hold two modes:

- a **mixture density** output, two components for stays and departs; or
- **classification plus conditional regression** — a departure probability, then the trajectory given
  each branch.

Either gives a well-posed target for windows that are genuinely ambiguous, and either would make the
per-group `beta_nll` unnecessary rather than merely tuned: the zero inflation that hack compensates
for *is* the stationary component of the mixture. That is a falsifiable prediction — if beta-NLL is
still needed after the output is made bimodal, the zero inflation was not about regime after all.

### The prediction was tested on 19 September 2026, and it failed

Three Port_Service pilot runs, `bilstm_attention`, 15 epochs, release 0918, identical cohorts.

| Run | val-loss jitter | test ADE | val_loss |
|---|---:|---:|---:|
| A · Gaussian, beta=0.5 | **0.275** | 728.8 | 2.291 |
| B · Gaussian, beta=0 | 0.729 | 717.8 | 2.972 |
| C · Mixture K=2, beta=0 | **0.726** | 751.0 | −1.850 |

**The mixture did not replace beta-NLL.** Its jitter is 0.726 against plain Gaussian's 0.729 — no
improvement whatsoever. Mixture-mean ADE was also worse, which was expected and means little on its
own, since the mean of a bimodal prediction is the between-modes point this design exists to avoid.

The large NLL gain is real — B and C are on the same scale, verified to 5e-07 by a one-component
mixture reproducing the Gaussian loss — but **it is not evidence of bimodal modelling**. Inspecting
the trained components:

- both sit at the **origin**: median 60-minute predicted displacement 0.2 m and 0.0 m
- one is broad (sigma about 4 km), the other a near-delta at the 1 m sigma floor
- mixture weights are **0.462 against 0.464** for windows that stayed and windows that departed

So the model learned the *marginal* shape — a spike on the stationary mass plus a diffuse blob — and
nothing about *which* vessel departs. It earns its likelihood from the 78% of windows that move about
a metre, and represents departures as undirected diffusion.

**A pilot of 15 epochs on 8,192 windows with one seed is thin evidence about training stability**, and
P2.0 requires the corrected full run before any result here is attributable. The failure is recorded
now because the prediction was pre-registered, not because the experiment is conclusive.

What the failure does **not** show is that the target is unimodal. Measurement 2 above is a property
of the data, not of any model, and it stands. What it shows is that an *unsupervised* gate cannot find
the split. A supervised one can, comfortably: see `DEPARTURE_PREDICTABILITY.md`, where a classifier on
the same inputs reaches AUC 0.961 on the strictly-stationary Port_Service population. **That redirects
the design from a mixture density to P2.0's design C, and it is the reason the mixture route is not
recommended.**

**Still open.** The under-way threshold. `under_way_knots` defaults to 0.5 through a `cfg.get` and is
not in `config()`, with no recorded justification. It does very different work per group — the share
labelled under way moves from 42.2% to 19.1% for Research_Offshore between 0.2 and 2.0 kn, but only
27.7% to 24.1% for Tanker. A single global threshold is a strong assumption where the curve is steep.

## Caveats

- Measurement 1 uses 8 shards per group, measurements 3 and 4 use 6, and one architecture. The effect
  sizes are far too large for sampling to explain, but **re-run at full scale before publication**.
- The two measurements use different releases, for the provenance reason given above. The regime
  structure should be confirmed on a single release once models exist for 0918.
- "Moves" is defined by mean implied speed over the full 60 minutes. A vessel that departs at minute 55
  is labelled as staying. That makes the transition share reported here a **lower bound**.
- Fishing is excluded from conclusions: 136 not-under-way windows in the sample, and 2 training vessels
  in the release.

## Why this matters beyond this dataset

Per-vessel-type evaluation is standard in AIS trajectory forecasting. This shows a group-level metric
can be reporting **fleet composition and the frequency of regime transitions** rather than the
difficulty of the forecasting problem. A group scoring worse than a constant-position baseline is not
necessarily harder to predict; it may simply contain more windows where the vessel's state changes
during the horizon, which no unimodal predictor can represent.

That is testable elsewhere and does not depend on the environmental work.
