# Joining environmental data to the 5–60-minute windows

Waves and currents from the Copernicus NW Shelf model, wind and pressure from the Met Office.

Plan of 17 September 2026. **Status:** both environmental sources are downloaded and verified, and
Step 0 has run. No join code is written yet, and the work remains sequenced behind the stability
rerun and regime separation.

Source: the Copernicus Marine NW Shelf family, produced by the Met Office — an AMM15 coupled
hydrodynamic-wave system with tides. **Resolutions differ by dataset**; see "Datasets, resolutions and
coverage" below for the verified IDs rather than relying on the headline 1.5 km figure.

## In plain terms

Picture walking along a moving walkway at an airport. Film only your feet and you are strolling;
film from outside and you are flying past. Same legs, different ground. **AIS is the camera
outside** — it records where the ship ended up, your walking plus the belt, but never says how fast
the belt was moving. For a ship the belt is the tide, and the waves are the bumps that slow it and
nudge it off course.

The three datasets never reference each other. There is no shared identifier, no ship name in the
weather file. What they share is that **every AIS record happened at a place and at a moment**, and
the sea model describes every place at every moment — so they are matched on where and when, the
same way you would look up the weather on the afternoon a photograph was taken.

A plain-language version with diagrams is published at
https://claude.ai/artifact/N8Gm5iUS22GST7uaVkJD5h

## Why these three datasets belong together

### A ground track is a sum, not a measurement of intent

AIS records motion **over the ground**. A master controls motion **through the water**. They differ
by the water itself:

```
V_ground  =  V_through-water  +  V_current
             ^                   ^
             what the master     what the sea
             commands            does to the hull
```

and `V_through-water` is itself degraded and deflected by waves and wind — added resistance slows
it, the sea state deflects it. Every position in the release is therefore a **sum of a decision and
an environment**, and the model sees only the sum. It is asked to predict 60 minutes of that sum
while observing neither term separately.

### The three layers are each other's boundary conditions

```
Met Office atmosphere  ->  NW Shelf ocean model  ->  AIS
wind stress, pressure      waves + tidal currents    vessel ground track
   energy input            energy in the water       floating body response
```

Not three loosely related sources but one causal chain: wind is the energy input, the NW Shelf model
is how that energy plus the tide and bathymetry becomes a sea state, and the vessel is a body
responding to that sea state under human control. The Met Office runs it as a **coupled**
hydrodynamic-wave system precisely because waves and currents modify each other.

### The arrays already measure the effect, not the cause

`X` holds COG (features 3–4) and heading (5–6) **separately**. The angle between them is the drift
angle — the measured signature of the vessel being set sideways by water and air. A 5° divergence
means two entirely different futures:

- **a rudder input** — the vessel is turning; expect the turn to continue
- **a 2-knot cross-tide** — the vessel is holding course while being set sideways; expect the set to
  continue, then *reverse* when the tide turns

The model cannot distinguish these. Environmental covariates make the decomposition identifiable.
That is the scientific justification, and it is considerably stronger than "more context improves
accuracy".

### Timescales: the two sources do different jobs

| | Cycle | Within the 60-minute horizon |
|---|---|---|
| Tidal current | ~12.4 h semidiurnal | **changes materially**, can reverse near slack water |
| Wind-sea | hours | slowly varying |
| Swell | days | essentially constant |

Currents are a **dynamic forcing** that evolves inside the prediction window; waves are a
**quasi-static modifier** of how efficiently the vessel moves. Different roles, so both belong.

### Magnitude: the forcing is comparable to, or larger than, the signal

Mean net ground displacement over 60 minutes, from the constant-position baseline, in knots:

| Group | Net displacement | Skill |
|---|---:|---:|
| Fishing | 4.35 kn | 0.83 |
| Cargo | 4.01 kn | 0.29 |
| Tanker | 1.79 kn | 0.31 |
| Unknown | 1.62 kn | 0.78 |
| Passenger | 1.44 kn | 0.74 |
| Research Offshore | 0.82 kn | 0.98 |
| **Port Service** | **0.63 kn** | **1.04** |

**Superseded 17 September 2026 — but note carefully which half.** An earlier version argued that
Mersey tidal streams of 2–3 knots move Port_Service vessels several times more than they move
themselves. Two separate claims are bundled there, and they fared differently:

- **The magnitude claim holds.** Measured on the downloaded 1.5 km field: peak current **1.77 m/s
  (3.4 kn)**, 95th percentile 0.84 m/s (1.6 kn). An intermediate retraction of this figure was made on
  7 km evidence showing only 1.3 kn — that product smooths the peaks, and the retraction was wrong.
- **The mechanism claim does not hold.** Direct measurement on the release withdraws it:
  - Fitting an M2 harmonic (12.4206 h) to 60-minute displacements gives Port_Service an amplitude of
    121 m against an RMS displacement of 2,063 m — 5.8%. Cargo gives 30.4%, the opposite of the
    expected ordering, and almost certainly reflects **tidal scheduling** (transits timed to the
    tide) rather than hydrodynamic set.
  - Restricting to windows where the vessel is not under way, so that any displacement must be
    drift: Port_Service has a **median displacement of one metre over sixty minutes**. Those vessels
    are moored alongside, not adrift. A moored vessel does not follow the tide.

What the measurement shows instead is that the low-skill groups are **mixtures**. For Port_Service,
68.3% of test windows are not under way and carry 19.7% of the error; the 31.7% under way carry 80.3%.
The ordering of skill by net displacement is real, but it reflects regime mixture rather than
environmental forcing.

**The environmental hypothesis survives only in a narrower form:** tidal currents may help the
**under-way subset**, not the groups as a whole. See the revised sequencing below.

(Net displacement is not speed — a vessel can travel far and return — but it is the right scale for
"how far from where it started", which is what the task predicts.)

### The covariates are legitimately forecastable

Tides are deterministic, computable years ahead from harmonic constants; wave and wind forecasts are
skilful at 1–7 days, far beyond a 60-minute horizon. At prediction time these inputs would genuinely
be available. That separates them from covariates such as neighbouring-vessel behaviour, which would
have to be forecast as hard as the target itself.

### Limits of the explanation

- It accounts for **environmental** deviation, not **intent**. A decision to alter course for
  traffic or a berth change stays unexplained.
- The grid is coarse relative to dock-scale motion: 1.5 km at best, and 7 km for the historical
  reanalysis, where the whole footprint box holds only 32 wet cells and the Mersey channel is not
  resolved at all. See step 0 below.
- Establishing the mechanism requires improvement to **concentrate where the physics predicts**, not
  merely a global ADE drop. Improvement in the wrong places means the feature is correlated, not causal.
- **Expect a small effect.** The only quantified ablation in the literature (EnvShip) reports 8.2%
  error reduction at a 10-minute horizon but **2.2% at 60 minutes**, and finds environmental context
  actively unhelpful in open water (100.4 m against 92.8 m for a trajectory-only model). Sixty minutes
  is our horizon. The earlier expectation that waves would lift the open-water groups runs against that
  evidence and is withdrawn.
- Two reasons our case may still differ: EnvShip's environmental representation is largely **coastline
  geometry** (a signed distance field and binary rasters), not hydrodynamics; and none of its four
  regions — Denmark, the United States, Greece, Norway — is macro-tidal. The specific hypothesis about
  tidal currents and under-way vessels in a macro-tidal estuary remains untested.

The dataset paper lists "no weather, tide or neighbouring-vessel context" as a limitation; this
closes part of it.

## Measured scope

| | |
|---|---|
| Window span | 2023-08-22 → 2026-09-11 (26,781 hours) |
| Vessel footprint | lat 53.286–53.764, lon −3.528 to −2.751 |
| Distinct wave-grid cells occupied | **151** in a 73k-window sample |
| Unique `(cell, hour)` pairs | 8.6% of windows — heavy redundancy |

The footprint is far smaller than the study box (52.8–54.2 N, 4.2–2.2 W), so the download is a
small subset, not the whole shelf.

## Datasets, resolutions and coverage — verified 17 September 2026

Read from the live catalogue with `copernicusmarine describe`, not from product pages. **The product
pages quote AMM15 at 1.5 km, but that resolution does not apply to every dataset in the family** — the
multi-year physics reanalysis is 7 km. An earlier draft of this plan said 1.5 km throughout and was
wrong.

| Purpose | Dataset ID | Grid | Cadence | Covers |
|---|---|---|---|---|
| Currents, high resolution | `cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT15M-i` | **1.5 km** | **15 min** | 2024-08-04 → 2026-09-23 |
| Currents, hourly variant | `cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i` | 1.5 km | 1 h | as above |
| Currents, historical | `cmems_mod_nws_phy-uv_my_7km-2D_PT1H-i` | **7 km** | 1 h | reanalysis, back to 1993 |
| Sea surface height | `cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT15M-i` | 1.5 km | 15 min | as anfc |
| Waves, forecast | `cmems_mod_nws_wav_anfc_1.5km_PT1H-i` | 1.5 km | 1 h | anfc archive |
| Waves, reanalysis | `MetO-NWS-WAV-RAN` | 1.5 km | 1 h | → 2026-04-30 |

Variables in the current datasets are `uo` (eastward) and `vo` (northward), in m/s.

### Coverage against the splits

The 1.5 km archive begins **4 August 2024**, which leaves a 347-day gap at the start of the AIS span —
but that gap falls entirely inside training:

| Split | Covered at 1.5 km |
|---|---:|
| train | 53% |
| validation | **100%** |
| calibration | **100%** |
| test | **100%** |

**Every split that produces a reported number is fully covered at the higher resolution.** Only the
earlier half of training falls back to 7 km.

### Resolution matters more than the numbers suggest

Measured on the actual footprint box (lat 53.2–53.8, lon −3.6 to −2.7):

- **7 km**: 9 × 8 = 72 cells, of which **32 are wet**. The Mersey channel is not resolved at all.
- **1.5 km**: roughly 1,350 cells. The vessel footprint occupied 151 of them.

A twentyfold difference in spatial sampling and four times the temporal sampling.

### Recommended choice

Use **1.5 km 15-minute currents from 2024-08-04 onward**, and train on that subset alone rather than
mixing resolutions. Mixing 7 km and 1.5 km forcing inside one training set introduces a covariate
shift indistinguishable from the environmental effect being measured. It costs 47% of training windows;
with 16 million of them that is affordable, and homogeneous forcing is worth more than volume here.

Carry two caveats: the 1.5 km product is *analysis-forecast*, so earlier portions may mix analysis with
forecast fields; and a deployed system would have forecast rather than analysis values, which is the
same causality note recorded elsewhere in this plan.

### The tidal signal is confirmed present

Verified on a real two-day download (2024-06-15 to 06-17, 7 km product, 38 KB):

- A **single M2 harmonic explains 97.9%** of eastward current variance and 84.9% of northward
- Mean M2 amplitude 0.329 m/s east; residual non-tidal flow only 0.025 m/s
- Peak speed in the box 0.68 m/s (1.3 kn) over that sample

The product is overwhelmingly tidal, so the null-result-for-the-wrong-reason risk is closed for
currents. **Superseded 17 September 2026 for the peak-speed figure only.** The 1.3 kn quoted here is a
7 km-product number, and that product smooths the peaks; the 1.5 km download gives a peak of 1.75 m/s
(3.4 kn) with a 95th percentile of 0.84 m/s (1.6 kn). An intermediate retraction of the 2–3 kn figure
was made on this 7 km evidence and was itself wrong — see the corrected figures earlier in this plan.
The M2 variance shares above are unaffected.

### Working commands

The toolbox lives in an isolated environment, because `copernicusmarine` requires numpy ≥ 2.1 while
TensorFlow 2.16.2 requires numpy < 2.0. Installing it into the training environment breaks TensorFlow.

```bash
~/.venvs/cmems/bin/copernicusmarine describe --contains NWSHELF --disable-progress-bar
~/.venvs/cmems/bin/copernicusmarine subset \
  -i cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT15M-i \
  -x -3.6 -X -2.7 -y 53.2 -Y 53.8 \
  -t "2024-08-04T00:00:00" -T "2026-09-11T23:59:59" \
  -v uo -v vo -o env_cache -f nws_cur_1p5km.nc --disable-progress-bar
```

Credentials are stored at `~/.copernicusmarine/` by `copernicusmarine login` and are picked up by the
venv copy. Read the resulting NetCDF from the training environment, where `xarray` and `netCDF4` remain
installed against numpy 1.26.4.

## Step 0 — wet-cell coverage: RUN 17 September 2026

**Result, against the real download** (`env_cache/cur_1p5km.nc`, 1.5 km hourly currents, 631 wet
cells in the footprint box) and the actual window origins from the test split:

| Group | On a wet cell | Within one cell of wet |
|---|---:|---:|
| Research_Offshore | **98.7%** | 99.1% |
| Cargo | 70.7% | 87.9% |
| Fishing | 69.0% | 70.6% |
| Port_Service | **36.9%** | 92.1% |
| Unknown | 35.6% | 92.7% |
| Passenger | **14.7%** | 90.0% |
| Tanker | **11.1%** | 87.7% |

**Exact-cell lookup fails for most groups.** Tanker at 11% and Passenger at 15% are in the river and
at berth, where a 1.5 km ocean grid simply has no water — the model's coastline does not extend into
the dock estate. Only Research_Offshore, which operates offshore, is well served by exact lookup.

The right-hand column is the decisive one: **roughly 90% of windows lie within one cell of modelled
water.** The problem is representational, not fatal.

### Decision — nearest wet cell, with the distance recorded

Do not use exact-cell lookup, and do not silently substitute a neighbour either.

1. Take the nearest **wet** cell within a bounded search radius, not the containing cell.
2. Record `current_cell_distance_m` per window — how far the sampled water was from the vessel.
3. Record a **per-source** validity mask — `current_valid`, `wave_valid` — false when no wet cell of
   *that* model lies within the radius. One shared flag is wrong; see the mask asymmetry below.
4. **Report results stratified by that distance.** A window sampled 300 m from its vessel is a
   different proposition from one sampled 4 km away, and averaging them hides exactly the weakness a
   reviewer would probe.

The radius is a declared design choice, not an optimum. Start at one cell (~1.5 km) and report how
many windows it excludes per group; widen only with the exclusion count published alongside.

This also gives a natural falsification test: **if the environmental effect is real, it should be
strongest where `current_cell_distance_m` is smallest.** An effect that does not decay with sampling
distance is not describing the water the vessel is actually in.

### Implementation warning — grid indexing

The first run of this analysis used `np.searchsorted` for grid indexing, which is off by one against
nearest-neighbour. That systematic one-cell shift pushed coastal vessels onto land and reported Tanker
at 7.8% rather than 11.1%. It was caught only because 7.8% was implausible for deep-water vessels.

Use `rint((v - v0) / dv)` for cell indices, and assert against a known point. **A grid-indexing error
of this kind looks like a physical finding rather than a bug**, which is what makes it dangerous — it
would have been reported as "the model cannot see these vessels" instead of "my indexing is wrong".

## Measured on the real downloads — 17 September 2026

All three sources are on disk and verified, 1.5 km hourly throughout:

| File | Size | Variables | Span | Steps |
|---|---:|---|---|---:|
| `env_cache/cur_1p5km.nc` | 92 MB | `uo`, `vo` | 2024-08-04 → 2026-09-11 | 18,455 |
| `env_cache/wav_1p5km.nc` | 275 MB | `VHM0`, `VTM02`, `VMDR`, `VHM0_SW1`, `VSDX`, `VSDY` | 2024-08-06 → 2026-09-11 | 18,408 |
| `env_cache/ssh_1p5km.nc` | 46 MB | `zos` | 2024-08-04 → 2026-09-11 | 18,455 |

Significant wave height in the box: median 0.50 m, 95th percentile 1.81 m, max 5.62 m — plausible for
a shallow bay sheltered by Ireland and Wales. Sea surface height ranges over **10.41 m** across the
record, the macro-tidal signature this project's setting argument depends on.

### The three sources separate cleanly by physics

Variance explained by a single M2 harmonic:

| Source | M2 variance explained |
|---|---:|
| Sea surface height | **80.7%** |
| Currents | **78.3%** |
| Significant wave height | **0.9%** |

Tide, tide, and storm. This is a real verification rather than a formality: a strongly tidal wave
field would have meant a wrong dataset or a mislabelled variable, and a weakly tidal current field
would have meant a tideless product. It also justifies keeping `CUR`, `SSH` and `WAV` as separate
ablation blocks — the first two are not proxies for the third.

### Mask asymmetry — it is the velocity field that is restricted, not the wave grid

**Revised after adding sea surface height.** An earlier version of this section, written when only
currents and waves were on disk, concluded that "the wave grid reaches further inshore". The
three-way comparison shows that framing is wrong.

| Source | Variable | Wet cells in the footprint box |
|---|---|---:|
| Sea surface height | `zos` | **670** |
| Waves | `VHM0` | **670** — identical to `zos` |
| Currents | `uo`, `vo` | **631** — a strict subset |

`zos` and the wave field share an identical mask. It is specifically **`uo`/`vo` that are masked in
39 cells where `zos` is defined** — and both are outputs of the same physics model, so this is not a
difference between models. The likely cause is velocity being undefined in very shallow or
wetting-drying cells where sea level still is.

Every cell with currents has sea level and waves; 39 cells have sea level and waves but no currents.

### Consequence — `zos` may be the better tidal feature, not a supplement

This matters most exactly where coverage was worst. Current coverage of window origins was Tanker
11.1%, Passenger 14.7%, Port_Service 36.9% — the near-shore groups. **Sea surface height is available
in strictly more cells than currents**, so those groups may be servable by `zos` where `uo`/`vo` are
simply absent.

`zos` also encodes tidal phase in a single scalar. Sea level and tidal stream run roughly in
quadrature in an estuary — high water near slack, peak flow near mid-tide — so one number places a
window in the tidal cycle without needing a vector. That is more robust at cells where a 1.5 km
velocity field is questionable, and it is the quantity that governs lock-gate access, which is the
operational argument this project makes.

Measured on the download: M2 explains **80.7%** of `zos` variance, mean M2 amplitude **2.87 m**, and
the total range over the record is **10.41 m**. That last figure is a direct confirmation of the
macro-tidal setting the novelty claim rests on, measured rather than quoted.

Consequences for the sidecar:

- **A single shared validity flag is wrong.** Use `current_valid`, `wave_valid` and `ssh_valid`
  separately. In practice `wave_valid` and `ssh_valid` coincide, but do not rely on that holding for
  other spans or products — assert it instead.
- The nearest-wet-cell search must run **per source**, against that source's own mask, with
  `current_cell_distance_m`, `wave_cell_distance_m` and `ssh_cell_distance_m` recorded separately.
- Ablation runs are therefore not on identical cohorts. A `CUR`-only run has fewer usable windows than
  a `WAV`- or `SSH`-only run. **Compare on the intersection**, or the difference between blocks is
  partly a difference in population rather than in information.
- Consider an `SSH` block in the ablation in its own right, not folded into `CUR`. If sea level helps
  where currents are unavailable, that is a separate and useful finding.

### Start dates differ by two days

Currents begin 2024-08-04, waves 2024-08-06. Trivial in itself, but the builder must take the
intersection of available spans rather than assume alignment — an off-by-two-days slice would
silently misalign every window in the first two days.

### Licence and attribution

Copernicus Marine grants a worldwide, non-exclusive, royalty-free, perpetual licence. Commercial use,
redistribution and derived products are all permitted; new IP in modifications belongs to the licensee.
The obligation is attribution, with required wording:

> "This study has been conducted using E.U. Copernicus Marine Service Information; [insert DOIs]"

Record the dataset IDs **and their DOIs** in the run manifest alongside the existing provenance. Note
that this puts the environmental layer in a materially better position than the AIS data, whose
redistribution permission remains unresolved: a future release carrying joined environmental features
would be unambiguously redistributable on that side.


## Design decisions

**Sidecar, not a rebuild.** The release is frozen, checksummed and 6 GB. Do not regenerate it.
Write a sidecar Parquet keyed by the window identity `(segment_id, origin_time_ns)`, bucketed to
mirror `clean_tracks/bucket_NNN.parquet`, and join it in `iter_batches`. `KEYS` in
[workflow.py](workflow.py) already loads `segment_id`, `origin_time_ns` and `origin_xy_m`, so the
join key and the position are present in every batch with no shard changes.

**Per-window static context, not per-timestep.** Waves are hourly; the input window is 20 minutes.
Follow the existing `use_spatial_context` pattern in `model_input`, which repeats `origin_xy_m`
across the 20 steps — add a `use_wave_context` flag and extend the feature count from 18.

**Encode direction as sin/cos, never degrees** — consistent with how COG and heading are already
stored, and avoids the 359°→0° discontinuity.

**Carry wave direction relative to heading.** Absolute wave direction matters far less than whether
the vessel is in head, beam or following seas, which is what drives added resistance and
course-keeping. Compute `wave_dir − heading` and store its sin/cos alongside the absolute values.
This is the feature most likely to earn its place.

## Proposed variables

| Variable | Why |
|---|---|
| `VHM0` significant wave height | the primary forcing magnitude |
| `VTM02` mean period | distinguishes short chop from long swell at equal height |
| `VMDR` mean direction → sin/cos | absolute direction |
| relative direction (`VMDR − heading`) → sin/cos | head / beam / following seas |
| `VHM0_WW`, `VHM0_SW1` wind-sea and swell height | locally generated vs remote swell behave differently |
| `VSDX`, `VSDY` Stokes drift | direct lateral transport, in m/s like SOG |
| `wave_valid`, `current_valid` | per-source wetness — the two models disagree, see below |

## The product seam — handle explicitly

The wave **reanalysis covers 1980 → 30 April 2026**; the data runs to 11 September 2026. That
boundary falls **inside the test period** (March–September 2026), so the test split would be built
from two different products.

Store a `wave_source` column per window recording which product supplied it, and audit test metrics
either side of the seam. Do not quietly concatenate them.

**Currents have a seam too, handled differently.** The 1.5 km archive begins 2024-08-04, so a window
before that date can only be served at 7 km. Rather than concatenating two resolutions — which
introduces a covariate shift indistinguishable from the environmental effect — the recommendation
above is to *restrict* to 2024-08-04 onward and accept the loss of 47% of training windows. Record a
`current_source` column regardless, so the choice is visible in the artefact rather than implicit in a
date filter.

## Causality caveat to document

Reanalysis assimilates information from after the forecast origin, so training on it is optimistic
relative to a deployed system, which would have only a wave *forecast*. This is the same class of
issue the dataset paper already raises about offline cleaning using later observations. Document it,
and consider a later comparison against forecast fields.

Note also that both wave products list data assimilation as **"None"** — they are forced by
ERA5/Met Office atmospheric fields and global wave boundaries. The Sentinel assimilation applies to
the *physics* product (SST, sea level), not the wave one. Check what the supervisor intended before
writing "derived from Sentinel data" into a methods section.

## Technical data flow

### Where each source enters

```
 MySQL  ship_data_YYYYMMDD + vessel_data
   │
   │  prepare_training_data_validated.ipynb        [DONE — do not rerun]
   ▼
 clean_tracks/bucket_NNN.parquet                   58,313,391 observations
   │
   │  ais_forecasting_windows.py :: build_release  [DONE — release is frozen]
   ▼
 groups/<group>/<split>/*.npz                      20,599,126 windows
   │                                                 carries origin_xy_m,
   │                                                 origin_time_ns, segment_id
   │
   │                     ┌──────────────────────────────────────────┐
   │                     │  cmems_mod_nws_wav_anfc_1.5km_PT1H-i     │  waves
   │                     │  cmems_mod_nws_phy-cur_anfc_1.5km-2D_    │  currents
   │                     │      PT15M-i   (7km _my_ before 2024-08) │
   │                     │  CEDA   Met Office UKV  (or ERA5)        │  wind, pressure
   │                     └───────────────────┬──────────────────────┘
   │                                         │  subset to the footprint box,
   │                                         │  hourly, selected variables
   │                                         ▼
   │                               env_cache/*.nc                    [NEW]
   │                                         │
   │                                         │  build_env_sidecar.py [NEW]
   │                                         │  one pass per release bucket
   │                                         ▼
   └──────────────────────────────►  env/bucket_NNN.parquet          [NEW]
                                       key: (segment_id, origin_time_ns)
                                             │
              iter_batches()  ───────────────┘   left-join per shard
                     │
                     ▼
              model_input()    X : (N, 20, 18 + k)
                     │
                     ▼
                  model
```

Nothing upstream of the NPZ release changes. The release stays frozen and checksummed; the
environmental data arrives as a **sidecar** joined at load time.

### The lookup, precisely

For each window, once:

1. `origin_xy_m` (EPSG:32630) → lon/lat via `pyproj` — already a dependency
2. lon/lat → fractional grid indices on the 0.0135° × 0.0303° AMM15 grid
3. **Bilinear interpolation** over the four surrounding grid corners
4. **Linear interpolation in time** between the two bracketing hourly fields
5. Derive the relative-direction features against the window's final heading
6. Write one row keyed by `(segment_id, origin_time_ns)`

Steps 3 and 4 are the standard treatment — the Hierarchical Two-Stage paper (arXiv 2605.16442)
does exactly this. Cite it; do not present it as a contribution.

**Deduplicate first.** Only 8.6% of windows are unique `(cell, hour)` pairs, so resolving distinct
pairs and then broadcasting back is roughly a 12× saving on field lookups.

### Feature blocks — designed for ablation

Grouped so each source can be switched on independently. Current input is 18 (8 features + 8 masks
+ 2 spatial context); each block appends to that.

| Block | k | Features |
|---|---:|---|
| `CUR` | 4 | current east, current north, sin/cos of (current dir − heading) |
| `WAV` | 6 | Hs, mean period, sin/cos of (wave dir − heading), swell Hs, Stokes drift magnitude |
| `WND` | 4 | wind speed, sin/cos of (wind dir − heading), crosswind component |
| `mask` | 1 | whether the cell was wet and all three sources resolved |

All blocks enabled: **18 → 33 features**. Directions are never stored as degrees — always sin/cos,
matching how COG and heading are already encoded, and always **relative to the bow**, because head,
beam and following conditions act on a hull completely differently.

Waves and currents evolve slowly relative to a 20-minute window, so each block is a **per-window
static context** repeated across the 20 steps — exactly how `model_input` already handles
`origin_xy_m`.

### Code touchpoints

| File | Change |
|---|---|
| `build_env_sidecar.py` | **new** — steps 1–6 above, one Parquet per release bucket |
| `workflow.py` `iter_batches` | left-join the sidecar on `(segment_id, origin_time_ns)` |
| `workflow.py` `model_input` | append the enabled blocks; masked entries zero-filled as `X` already is |
| `workflow.py` `fit_scaler` | extend the context moments to the new dimensions, train split only |
| `workflow.py` `config` | `env_blocks=('CUR','WAV','WND')` and the cache path |
| `models.py` `build_model` | feature count 18 → 18 + k |

`KEYS` already loads `segment_id`, `origin_time_ns` and `origin_xy_m`, so the join key and the
position need no shard change.

### The ablation

The point is not that environmental data helps, but **which source helps which group**. Five runs
per group, identical seed and splits:

| Run | Blocks | Features | Tests |
|---|---|---:|---|
| A | none | 18 | reproduces the current result — a control |
| B | `CUR` | 23 | tidal currents alone |
| C | `WAV` | 25 | waves alone |
| D | `WND` | 24 | wind alone |
| F | `SSH` | 21 | **tidal phase alone** — added 20 September 2026 |
| E | all four | 39 | whether the sources are complementary or redundant |

**Arm F was missing from the original design and is a real gap, not a formality.** This plan already
argues that `zos` may be the better tidal feature: it is defined in 670 cells against the current
field's 631, it reaches 98.5% of test windows against `current_valid`'s 94.6%, and it encodes tidal
phase in a single scalar where currents need a vector. Folding it into `E` alone makes it impossible
to tell whether sea level or currents carried any tidal effect. It is also the **cheapest arm**, at
three features against the current block's five.

Declared before running: **B should lift Port_Service and Research_Offshore; C should lift Cargo,
Tanker and Passenger; D should be smallest and should concentrate in light, high-sided vessels.**

Declared for F on 20 September 2026, before it was run: **F should match or beat B on the near-berth
groups — Tanker, Passenger, Port_Service — because sea level is defined in cells where the velocity
field is not, and those are exactly the groups whose window origins fall in the dock estate. Where
`CUR` and `SSH` are both available, F should be the weaker of the two, since a scalar phase carries
less than a velocity vector. F beating B on the open-water groups would be the disconfirming
outcome: it would suggest the tidal signal is acting through something other than the water actually
moving the hull.**
Uniform improvement across all groups, or improvement in the wrong groups, means the features are
correlated rather than causal.

Run B and C first — D is the cheapest to add once the pipeline exists and the least likely to matter.

### Pre-registration — declared 20 September 2026, before the full-scale ablation

Recorded here rather than in a commit message so the git timestamp is the evidence that it preceded
the result.

**Reporting is split into two horizon bands: 5–30 minutes and 35–60 minutes.** Both are reported on
every run and neither may be dropped. `HORIZON_SPLIT_MIN` in `workflow.py` is 30; `evaluate()` writes
`ade_5_30_m` and `ade_35_60_m`, and the same split per regime.

**Why split.** The expected effect is a gradient, not a level. EnvShip measures 8.2% error reduction
at a 10-minute horizon against 2.2% at 60. Error grows roughly threefold across our range — Cargo
averages 970 m at 5–30 minutes and 2,590 m at 35–60 — so a single 12-horizon mean is dominated by the
long band and a short-horizon effect would be averaged away.

**Two comparisons, so correct for two.** Splitting doubles the opportunity for a spurious hit. Any
claim of significance must account for both bands. With six arms and two bands the comparison count
is larger again; state the correction used.

### The power problem, measured — and it is the binding constraint

Four seeds of the trajectory-only control, pilot scale, per group. **The control disagrees with itself
by more than the effect being hunted:**

| | 12-horizon CV | 5–30 min CV | 35–60 min CV |
|---|---:|---:|---:|
| Cargo | 8.57% | **11.09%** | 7.95% |
| Port_Service | 4.42% | **8.35%** | **3.41%** |

Cargo's control ADE ranged 1,473–1,780 m across seeds — a 21% swing from initialisation alone.

Runs per arm to detect a 2.2% effect at 80% power:

| Cell | Runs per arm |
|---|---:|
| Cargo, 5–30 min | 399 |
| Cargo, 35–60 min | 205 |
| Port_Service, 5–30 min | 226 |
| **Port_Service, 35–60 min** | **38** |

**Note the asymmetry before interpreting any null.** The short band is the *noisier* of the two, so a
null result there is uninformative — yet that is exactly where EnvShip found its largest effect. The
statistically strongest cell in this design is Port_Service at 35–60 minutes, which is where the
effect is expected to be smallest. Say so when reporting, rather than presenting a short-band null as
evidence of absence.

**Consequences, binding on the full run:**

- **"Identical seed" is not enough.** The comparison above holds the seed fixed and still varies by
  8–11%, because changing the input width changes the initialisation. Every arm needs **multiple
  seeds**, and differences must be reported with a spread, never as a single number.
- Pilot scale cannot answer the question at all. The full run — more data, more epochs — is not
  procedure here; it is what makes the variance small enough for the effect to be visible.
- Train on the **covered era alone** (2024-08-04 onward, 53–57% of training windows), or the
  environmental arms differ from the control in population as well as in features.

### A pilot ablation was already run, and it was uninformative

Declared for transparency, since the registration above is not blind: A–E were run on Cargo and
Port_Service at pilot scale on 19–20 September 2026. Every effect fell within 0.13–1.69 standard
deviations of the control's own seed-to-seed spread, including the one apparent improvement
(Cargo, `WND`, −1.1%, which is 0.13 sd and would need 871 runs per arm to establish). Cargo degraded
monotonically with feature count — 18 features 1,780 m, 23–25 features 1,811–1,926 m, 39 features
2,038 m — which is as consistent with four draws from one distribution as with a capacity cost.

**Nothing in that pilot is evidence for or against the environmental hypothesis**, and no prediction
above was adjusted in light of it.

### Met Office wind — access and consistency

| Route | Notes |
|---|---|
| **CEDA Archive** | Met Office data under the NERC–Met Office agreement; holds historical UKV at **1.5 km** over the UK interior, which includes Liverpool (the 2 km figure quoted previously is DataHub's, not UKV's native resolution). Best fit for a UK academic needing Aug 2023 – Sep 2026, and **co-located with JASMIN**, where this project already has an account — so it is a filesystem read rather than a download. Note a JASMIN login does not itself grant CEDA dataset access; that is a separate registration and licence acceptance per dataset group. |
| Met Office Weather DataHub | API, UK deterministic 2 km, free tier ~1 GB/month. Forecast-oriented; awkward for three years of history. |
| Open-Meteo UKMO API | Wraps Met Office output, free for non-commercial. Easiest to prototype against. |

**Consistency trade-off — revised 19 September 2026, and the revision matters.** The sentence
previously here read "the NW Shelf wave reanalysis is forced by ERA5", and concluded that ERA5 wind
is consistent with the wave field by construction. The qualifier is the problem: **the wave file
actually downloaded is the analysis-forecast product**, `cmems_mod_nws_wav_anfc_1.5km_PT1H-i`, not
the reanalysis. `build_env_sidecar.py` had dropped the qualifier entirely and stated the claim
generally, which is how a rationale about one product came to justify the use of another.

The forcing of the analysis-forecast product is **not verified**. The catalogue entry for
`NWSHELF_ANALYSISFORECAST_WAV_004_014` cites Bruciaferri et al. 2021 on the Met Office coupled
ocean-wave system, which points to Unified Model forcing rather than ERA5. **If that is right, the
consistency argument favours UKV and the choice recorded here was backwards.**

Resolve it from the Product User Manual / QUID for `NWSHELF_ANALYSISFORECAST_WAV_004_014`, which
names the atmospheric forcing explicitly. Do not write either choice into a paper before then — a
reviewer familiar with the NWS system would catch a backwards justification, and it is the kind of
error that discredits the surrounding argument rather than just the sentence.

ERA5 remains in use meanwhile for reasons independent of consistency: it is roughly 10 MB for this
footprint, needs no CEDA licence, and sits on a regular lat/lon grid the builder already indexes.
Resolution against consistency is still the trade — but note that at 0.25° ERA5 spans only about
**four cells across the whole fleet**, so it contributes a temporal signal and essentially no
spatial one.

### What wind adds that the ocean model does not

The wave field already encodes wind's effect on the *sea*. Wind is worth carrying separately for
four reasons: **windage** — aerodynamic force on the hull and superstructure above the waterline,
absent from any wave or current field and proportionally large for slow or high-sided vessels;
**lead time** — sea state lags wind by hours, so wind is the earlier signal inside a 60-minute
horizon; **surge** — pressure and wind setup raise or lower water level, which shifts the tidal
window; and **visibility** — fog drives speed reduction and pilotage suspension, a behavioural
covariate rather than a force.

## Implementation outline

1. **Access** — free Copernicus Marine account, then the `copernicusmarine` toolbox. Not currently
   installed; `netCDF4` is missing too, `xarray` is present. Credentials are the user's to set up.
2. **Subset and cache** — download the footprint bounding box only (round outward to
   lat 53.2–53.8, lon −3.6 to −2.7), hourly, selected variables, as NetCDF.
3. **Build the sidecar** — for each release bucket: inverse-project `origin_xy_m` from EPSG:32630
   to lat/lon with pyproj (already a dependency), snap to the wave grid, interpolate linearly
   between the bracketing hourly fields, derive the relative-direction features, and write
   `wave/bucket_NNN.parquet`. Deduplicate on `(cell, hour)` first — 8.6% uniqueness means roughly
   a 12× saving on field lookups.
4. **Loader** — join the sidecar in `iter_batches`, extend `model_input` and the feature count,
   and extend `fit_scaler` to cover the new context dimensions (train split only, as now).
5. **Provenance** — record the product identifiers, download date, variable list and sidecar
   checksums in the run manifest, matching how the release records its own inputs.

## Verification

- Wet-cell coverage by group, from step 0, reported before any training
- Spot-check a handful of windows against the raw NetCDF by hand
- Confirm sidecar row count equals the release window count, with no duplicate `(segment_id,
  origin_time_ns)` keys
- Confirm scaler moments are finite and the masked-fill convention still holds
- Retrain one group with and without wave context under a fixed seed, and compare

## Sequencing

**Revised 17 September 2026.** Two things now come before any of this work.

1. **Fix training stability at full scale.** With the loss still being corrected, no improvement from
   environmental features could be attributed.
2. **Separate the motion regimes.** 68.3% of Port_Service test windows are not under way and carry
   only 19.7% of the error. A 2% environmental effect — the magnitude the literature reports at a
   60-minute horizon — cannot be detected through a metric dominated by moored vessels. Regime
   separation needs no external data, follows directly from the measurement above, and is a
   prerequisite for testing the environmental hypothesis at all.

Only then is the join worth building, and then against the narrowed hypothesis: **tidal currents for
the under-way subset of Port_Service and Research_Offshore.** Waves are expected to give little at
sixty minutes; a null result there would replicate EnvShip rather than contradict it.

**Sourcing caution.** Most global reanalyses do not include tides. ERA5-derived current products —
including Open-Meteo's marine currents, which are ERA5-Ocean based — would give a null result for the
wrong reason. AMM15 explicitly is "a coupled hydrodynamic-wave model system with tides", which is why
`NWSHELF_MULTIYEAR_PHY_004_009` is the right source. Confirm the tidal component is present in
whatever is downloaded before trusting a negative result.
