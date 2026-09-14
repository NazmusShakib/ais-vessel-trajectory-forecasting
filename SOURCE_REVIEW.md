# How the four original notebooks were adapted

All four original notebooks were read. They are preserved in the project root. Their pre-work hashes are recorded in `qa/original_file_hashes.json`; code-only extracts used during review are under `qa/reviewed_*`.

| Original notebook | Retained idea | Replaced or corrected |
|---|---|---|
| `v2-constant_velocity_baseline_time_based.ipynb` | Constant position; last-step, mean-recent, median-recent, linear-fit and SOG/COG velocities; validation-selected baseline. | Reads the same NPZ windows as neural models; uses actual distinct observation times, metre coordinates and observation-age-adjusted extrapolation. All seven candidates remain explicit. |
| `v2-bilstm-ship_trajectory.ipynb` | Two BiLSTM layers, multi-head self-attention, direct multi-horizon prediction, optional Gaussian uncertainty. | Eight features plus masks and optional spatial context; twelve targets; new unit-consistent losses; train-only standardisation; calibration role; bounded streaming; measured region coverage. |
| `v2-bilstm-only-ship_trajectory.ipynb` | Separate BiLSTM-only model for comparison with attention. | Shares the new BiLSTM encoder and pooling design with the attention model so attention is the intentional architectural addition. The old final-state pooling is not copied as an additional simultaneous difference. |
| `v2-transformer-ship_trajectory_prediction.ipynb` | Positional encoding, self-attention, feed-forward/residual blocks and a direct decoder. | Shares the same data, scaling, target units, Gaussian head, selection/calibration rules and error calculations as the recurrent models. |

The older learned-model notebooks expect 26 engineered CSV features and five target horizons `[5,10,20,30,60]`. They contain degree-based losses and uncertainty calculations, their own time-ordered trip splits, moving-record filtering, and forward/backward course filling. Those preparation steps are not reused. The new release already supplies the chosen cohorts and supports all twelve five-minute horizons.

Some old diagram code contains schematic paths and fallback Cargo scores. The new notebooks generate charts from their own saved predictions and metric tables only. Existing saved models/scalers and previous paper metrics are never loaded as compatible new results.

The old BiLSTM-only versus attention comparison also differed in pooling and positional encoding. The new architecture pair uses matching pooling and avoids introducing a second positional-encoding block into the attention variant. Layer normalisation replaces batch normalisation in the recurrent encoder. These design changes are declared adaptations; results cannot be labelled an exact reproduction of old experiments.

The previous all-zero Keras timestep mask is replaced by explicit per-feature validity channels. A missing SOG or heading feature does not remove the whole observed timestep. Relative zero position and valid stationary measurements remain valid values.

The old log-standard-deviation clipping was expressed in degrees. Applying it unchanged in metres would cap standard deviation at `exp(4)`, approximately 54.6 metres. The new head uses softplus plus a configurable positive floor, with a fixed metre-to-model scale and explicit conversion back to metres.

A frozen temporal calibration partition is now used. This adds empirical per-horizon 2D prediction regions and coverage/area reporting, but does not establish confidence guarantees under dependent windows, distribution shift, or scarce vessel support. A diagonal Gaussian is also limited for branching route choices and cross-coordinate dependence. Ensembles or richer distributions require separately evaluated extensions.
