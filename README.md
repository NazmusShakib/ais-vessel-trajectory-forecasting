# New Liverpool AIS training workflow: 5–60 minutes

Four separate notebooks adapt the requested baseline, BiLSTM-attention, BiLSTM-only and Transformer approaches to the completed vessel-group dataset. **Everything created for this workflow is in this folder. Existing notebooks and the prepared dataset remain unchanged.**

The notebooks are supplied with small, clearly labelled Cargo smoke runs. These validate the software flow; they are not full trained research models and their errors must not be used as publication results.

## Start with these notebooks

| Notebook | Purpose |
|---|---|
| [01_constant_velocity_baselines.ipynb](01_constant_velocity_baselines.ipynb) | Constant position and six velocity variants; select the best baseline on validation ADE. |
| [02_bilstm_attention.ipynb](02_bilstm_attention.ipynb) | Two-layer BiLSTM with multi-head self-attention and a direct twelve-horizon decoder. |
| [03_bilstm_only.ipynb](03_bilstm_only.ipynb) | Matching BiLSTM encoder and pooling without the attention block. |
| [04_transformer.ipynb](04_transformer.ipynb) | Transformer encoder with positional encoding and the same input/output contract. |

Open the notebooks with a Python environment containing `requirements.txt`. The verification environment uses `/opt/anaconda3/bin/python` and TensorFlow 2.16.2. TensorFlow's supported installation varies by platform. No PyTorch or MySQL connection is required.

Open from this folder or the project root. Execute top to bottom. Training runs in a separate local Python process, using the notebook environment, with progress streamed into the cell; the verified configuration uses CPU execution and disables graph meta-optimization to avoid an installed Metal-plugin crash observed on this Mac. The computation lives in the two shared Python files, which must stay beside the notebooks. There are no dependencies on the old model notebooks at runtime.

## How training now works

1. Read the release manifest and shard index; check the release identity and reconcile index counts.
2. Select the group and retain the existing train, validation, calibration and test roles. Save the exact shard/row cohorts and verify selected shard checksums.
3. Fit feature standardisation **only on valid training entries**. Preserve missing-feature masks; do not backfill course or replace unknown values with fabricated measurements.
4. Train a separate neural model for each requested group. Choose checkpoints by validation loss, with early stopping and learning-rate reduction. Baselines have no fitted model weights and are selected by validation ADE.
5. Reload the selected checkpoint. Calibrate per-horizon 2D regions using calibration data only, after weights have been frozen.
6. Evaluate on the test cohort and save metre-based errors, region coverage and area, plus vessel/month breakdowns and actual prediction examples.

Every run gets a new timestamped directory under `runs/`; existing experiments are not overwritten. Failed runs are marked failed and retain their intermediate files. Automatic resumption of neural training is not implemented. Completed datasets are consumed directly: **do not rerun MySQL extraction**.

## Prepared data used
- must required files - /outputs/vessel_group_forecasting/data/full_5_60_20260918T072424Z
full_5_60_20260918T072424Z/
├── groups/          ← All vessel groups, splits and NPZ files
├── manifest.json    ← Dataset information
└── shard_index.csv  ← List of data files used by training

To upload data to server using scp
`scp "/path/on/mac/your_dataset.zip" YOUR_USERNAME@xfer-vm-01.jasmin.ac.uk:/your/jasmin/workspace/`

To upload data to server using rsync
`rsync -avP "/path/to/your_dataset.zip" YOUR_USERNAME@xfer-vm-01.jasmin.ac.uk:/your/jasmin/workspace/`

The pointer at `../outputs/vessel_group_forecasting/current_release.json` identifies:

- Observation source: `../data_collection/ais_validated_training_outputs/full_20260913T005019_998082Z/`
- Training release: `../outputs/vessel_group_forecasting/data/full_5_60_20260918T072424Z/`

The training release contains 20,599,126 windows in 5,101 NPZ shards. This folder references those files read-only rather than copying six gigabytes of arrays. Move/share the release too if running elsewhere, and set `AIS_DATASET_DIR` in `.env` to its new path. The direct dataset path bypasses the old release pointer; no MySQL connection or clean-track Parquet files are needed for training. A different release requires reviewing the explicit identity check in `release_info`.

| Role | UTC interval, end exclusive | Use |
|---|---|---|
| train | Available history before 1 September 2025 | Fit scaler and model weights |
| validation | 1 September–1 December 2025 | Select model/checkpoint and baseline variant |
| calibration | 1 December 2025–1 March 2026 | Adjust prediction-region sizes |
| test | 1 March–1 September 2026 | Exploratory evaluation after freezing the procedure |

The pipeline never re-splits overlapping windows. The same MMSI may appear in several periods; this is not an unseen-vessel experiment.

## Inputs, targets and units

Each source window has `X: (20,8)`, `X_mask: (20,8)` and `y: (12,2)`.

- Inputs are 20 one-minute grid timestamps spanning 19 minutes. Selected reports can repeat; their ages are explicit.
- Eight features: relative east/north metres; speed in metres/second; course sine/cosine; heading sine/cosine; observation age in seconds.
- The model receives the eight standardised values plus eight Boolean-mask channels. Invalid features are zero after standardisation, and valid zeros remain distinguishable.
- By default two standardised absolute UTM reference-position channels are repeated across the history, giving **18 model channels**. They supply known port-location context. Set `use_spatial_context=False` consistently for a 16-channel comparison.
- Targets are east/north displacements in metres at +5, +10, +15, …, +60 minutes. They are relative to the final **actual** observed input position, which may precede the nominal origin by up to 180 seconds.
- Neural targets use a fixed 1,000-metre scale internally. Output means and standard deviations are converted back to metres before calibration, evaluation and plotting. There is no fitted target scaler.
- The default Gaussian head predicts two means and two positive standard deviations per horizon, with softplus and a one-metre floor. It models diagonal conditional covariance; it does not represent multiple distinct route modes or all model uncertainty.
- Deterministic mode uses Huber loss and residual-calibrated circular regions. Use the same output mode across neural models for an architecture comparison.

Velocity baselines use distinct actual report times to avoid treating repeated grid selections as zero-speed measurements. Extrapolation includes report age. SOG/COG is projected from true bearing into the dataset's UTM coordinates. Missing SOG/COG falls back to last positional velocity; if that cannot be estimated, zero velocity is used. No speed cap is applied to hide errors.

## Smoke, pilot and full modes

Edit the configuration cell near the top of each notebook:

```python
MODE = 'pilot'       # 'smoke', 'pilot', or 'full'
GROUPS = ['Cargo', 'Tanker']
CFG = config(MODE, GROUPS)
```

| Mode | Selected data per group and role | Neural training |
|---|---|---|
| smoke | Up to 2 randomly selected shards, up to 64 rows each | Width 16, 1 epoch |
| pilot | Up to 32 randomly selected shards, up to 256 rows each | Width 128, up to 15 epochs |
| full | Every shard and every row in each selected group/role | Width 128, up to 100 epochs |

`full` means all examples **for the listed groups**. Use `--groups all` on the CLI or `GROUPS=['all']` in a notebook to select all seven groups. Supported names are Cargo, Tanker, Passenger, Port_Service, Research_Offshore, Fishing and Unknown. There are no silently borrowed validation/calibration examples for a group with missing roles.

Subset sampling is reproducible and shared across model families. It is not a representative seasonal or vessel-balanced sampling design. Full training traverses each training window once per epoch, shuffling shards and rows within each shard. It uses uniform window weighting; frequent vessels can still dominate. Vessel-balanced weighting is an explicit future experiment, not an implemented safeguard.

To run without a notebook, from this folder:

```bash
python run_training.py baselines --mode pilot --groups Cargo Tanker
python run_training.py bilstm_attention --mode full --groups Cargo Tanker
python run_training.py bilstm_only --mode full --groups Cargo Tanker
python run_training.py transformer --mode full --groups Cargo Tanker
```

Add `--deterministic` for a deterministic neural experiment or `--epochs 30` to override its epoch budget. See the hardware and path settings below before using a cluster. Full runs can be expensive on a CPU. The workflow keeps memory bounded by a source shard and minibatches; calibration scores are written to an on-disk array. It rereads compressed shards each epoch rather than materialising the whole dataset in memory.

## Portable hardware and paths (.env)

The supplied `.env` works with the current local folder layout. `.env.example` is a commented template for another computer. The script reads `.env` automatically from beside `run_training.py`; do not source it as shell code. No additional dotenv package is required.

```dotenv
AIS_DATASET_DIR=/your/storage/full_5_60_20260918T072424Z
AIS_DEVICE=auto
AIS_RUNS_DIR=/your/storage/ais_experiments
AIS_JOBS_DIR=/your/storage/ais_worker_logs
AIS_BATCH_SIZE=64
```

| Setting | Meaning |
|---|---|
| `AIS_DATASET_DIR` | Prepared release directory containing `manifest.json`, `shard_index.csv` and all NPZ shards in their existing relative layout. This is the dataset path, not a MySQL database. |
| `AIS_DEVICE` | `cpu`, `gpu` or `auto`; see selection rules below. |
| `AIS_RUNS_DIR` | Parent folder for timestamped runs, checkpoints, calibration and evaluation. Defaults to `runs` beside the script. |
| `AIS_JOBS_DIR` | Parent folder for notebook worker requests and logs. Defaults to `jobs` beside the script; CLI training does not use a worker log. |
| `AIS_BATCH_SIZE` | Positive batch size; default 64. |
| `AIS_RELEASE_POINTER` | Optional JSON pointer used only when no direct dataset path is supplied. Its `output` path must resolve on the destination computer. |

Absolute paths are simplest on a cluster. Relative dataset/output paths always start at the **training folder**, regardless of the terminal working directory or `.env` location. A relative `output` inside a pointer JSON starts at that JSON's directory. `--env-file` itself is resolved from the terminal working directory. Quote values containing spaces or `#`; use literal paths, without shell variables. `~` is supported. Unknown `.env` keys and invalid configuration values are rejected.

Priority is **explicit command/notebook arguments → exported AIS_* environment variables → .env → defaults**. The loader does not overwrite process environment variables. It rereads the file whenever `config(...)` is called. `--data-dir`, `--runs-dir`, `--jobs-dir`, `--batch-size` and `--device` provide command-line overrides. Set `--env-file /path/to/cluster.env` to choose another settings file.

- `cpu` hides TensorFlow GPUs. On macOS it also retains the verified Metal graph-optimizer workaround.
- `gpu` requires a detected GPU and a successful small GPU matrix operation before training. It fails instead of silently accepting CPU execution if that check fails.
- `auto` selects CPU on macOS because of the observed Metal issue; on other platforms it selects GPU if TensorFlow detects one, otherwise CPU. A detected but unusable GPU raises an error; use `cpu` explicitly if needed.

Neural runs print the resolved device and save it in `runtime.json` and `run.json`. Baselines use NumPy CPU calculations, so `--device` only affects neural models. GPU mode uses the first TensorFlow-visible GPU and preserves the scheduler's `CUDA_VISIBLE_DEVICES`; it does not combine multiple GPUs. GPU memory grows as needed on non-macOS platforms. Some operations may still run on CPU. This follows TensorFlow's [GPU configuration guidance](https://www.tensorflow.org/guide/gpu); the cluster needs a compatible [TensorFlow installation](https://www.tensorflow.org/install/pip). The local requirements file records the tested Mac environment, not a verified cluster CUDA environment.

Check the configuration **inside the allocated cluster job**, so the check sees its GPU:

```bash
python run_training.py bilstm_attention --mode full --groups all --device gpu --check-config
```

This reads release metadata, reports selected counts and checks TensorFlow hardware without fitting a model or scanning all shard checksums. A normal training run still verifies selected shard checksums. Then train using either explicit hardware selection:

```bash
python run_training.py bilstm_attention --mode full --groups all --device gpu
python run_training.py bilstm_attention --mode full --groups all --device cpu
```

Or rely on the `.env` device setting:

```bash
python run_training.py bilstm_attention --mode full --groups all
```

The existing notebooks also read `.env` through `CFG=config(MODE,GROUPS)`. Override with `CFG=config(MODE,GROUPS,device='gpu',release='/your/dataset',runs_dir='/your/results')` if desired. Their existing saved outputs are historical small CPU verification runs. Rerun from the top after changing settings; device changes require a new training worker or Python process.

Keep the full prepared release and this training folder together when transferring; the observation-source folder and MySQL are not runtime dependencies. Configure persistent writable result storage. Small Matplotlib/Python caches may still be written inside the training folder. Submit these commands through the university's job template with the corresponding CPU/GPU resource allocation; this launcher does not submit Slurm jobs. Groups run sequentially, with separate models and all existing data roles preserved. Automatic checkpoint resumption is still not implemented.


## Comparing runs fairly

Use identical group lists, seeds, modes, feature/context settings, output modes, and training/tuning budgets. Parameter counts are recorded; equal width does not imply equal parameter count. The BiLSTM-only variant here shares the attention model's pooling, unlike the older notebook's last-state pooling, to make the attention comparison cleaner. This is an adapted architecture comparison, not an exact reproduction of the old code.

```python
from workflow import compare_runs
comparison = compare_runs([
    'runs/<baseline_run>',
    'runs/<bilstm_attention_run>',
    'runs/<bilstm_only_run>',
    'runs/<transformer_run>',
])
```

`compare_runs` rejects mismatched release identities, modes, output/context settings or selected cohorts. It does not automatically establish matched tuning effort. For baselines, `selected_baseline=True` identifies the validation-selected candidate; do not select the strongest-looking test candidate retrospectively.

Per-horizon ADE uses all twelve horizons; FDE is the error at 60 minutes. These new ADE values cannot be directly compared with the old five-horizon ADE averages or older differently selected datasets.

## Prediction regions and their limits

After fitting, calibration computes a separate radial residual quantile for each horizon. Gaussian models obtain axis-aligned ellipses scaled by predicted standard deviations; deterministic models and baselines obtain circles. The empirical quantile uses rank `ceil((n+1) * nominal_coverage)` and rejects insufficient calibration support.

These are **empirically adjusted per-horizon 2D regions**. The nominal 90% is a target, not a proven reliability level for every vessel. Overlapping windows and temporal shift undermine ordinary exchangeability assumptions. The workflow reports actual test coverage and mean area instead of calling a ±2σ coordinate band a calibrated trajectory confidence region. It does not provide simultaneous whole-trajectory coverage or dependence-aware uncertainty bounds on its metrics.

Known dataset limitations remain: Fishing has only two contributing training MMSIs; Research_Offshore includes ambiguous inherited labels; Unknown is heterogeneous; about 98.64% of target labels are interpolated; March supplies 84.94% of test windows, while August has one. Source cleaning is offline and can use later observations. Review these issues before strong specialist, seasonal or online-deployment claims.

The requested four approaches compare model families within vessel groups. Shared and type-conditioned models, multiple training seeds, matched-budget tuning, dependence-aware confidence intervals and future independent confirmation remain additional research tasks.

## What is saved

Each `<AIS_RUNS_DIR>/<mode>_<architecture>_<timestamp>/` contains:

- `run.json`: configuration, code fingerprints, source identity and completion/failure status.
- `comparison.csv`: group/model test summary, mode and exact evaluation-cohort hash.
- `<group>/cohorts.json`: exact selected shards and row indices for all four roles.
- Neural groups: `scaler.json`, `model_contract.json`, `architecture.txt`, `best_model.keras`, `training_history.csv`, `history.json`.
- `calibration.json` and `calibration_scores.npy`: empirical adjustment and saved calibration scores.
- `validation_metrics.json`, `validation_horizons.csv`, `test_metrics.json`, `test_horizons.csv`.
- `*_by_vessel.csv`, `*_by_month.csv`: per-horizon subgroup results and counts.
- `test_examples.npz`: three real selected examples with predictions and source identity; used by the plots.
- `*_errors_coverage.png`, `*_example.png`: plots generated from that run's outputs.
- Baseline groups store equivalent evaluation artifacts in candidate subfolders, plus `selected_baseline.json` and `baseline_validation_comparison.csv`.

Sample exports and charts are illustrative. No schematic trajectories or hard-coded old Cargo scores are used as new predictions. Model reload always uses its own saved scaler and configuration in the training flow.

## Supporting files

- `workflow.py`: loading, cohort selection, standardisation, baselines, calibration, metrics, orchestration and comparison.
- `models.py`: neural architectures, TensorFlow dataset stream, fitting and checkpoint reload.
- `run_training.py`: command-line entry point, hardware/path overrides and configuration preflight.
- `.env` / `.env.example`: active local settings and a portable template.
- `settings.py` / `runtime_control.py`: environment configuration and TensorFlow device selection.
- `test_settings.py`: path, precedence, group selection and hardware-policy checks.
- `worker.py` and `jobs/`: process-isolated notebook execution, requests and logs.
- `SOURCE_REVIEW.md`: what was retained and changed from each original notebook.
- `test_workflow.py`: focused tests for data handling, age-adjusted baselines, masks and quantiles.
- `requirements.txt`: versions installed in the verification environment.
- `qa/`: notebook execution/HTML previews, source-review extracts, original-file hashes, test reports and verification evidence.
- `.cache/`: local runtime caches; not required for handoff.

Copy this folder (including hidden `.env`, or recreate it from `.env.example`) and the referenced prepared release for training elsewhere. The original project notebooks and source data remain unchanged; configured experiment outputs may now be outside this folder.

## Verification completed

All four delivered notebooks executed successfully on the same small Cargo cohorts: 128 training, 128 validation, 128 calibration and 90 test windows. Each neural model ran for one epoch with a small verification architecture. Saving, reloading, calibration and evaluation completed. **Full dataset training has not been run; these example results do not establish model accuracy.**

Ten focused workflow tests passed. Six neural contract checks covered all three architectures in deterministic and probabilistic modes, including save/reload consistency. All eight final saved charts were visually inspected for legibility and clipping. Notebook HTML previews were generated, but their full page layout was not visually inspected.

See `qa/latest_verified_runs.json` for the four completed runs, `qa/notebook_execution_checks.json` for execution checks, `qa/neural_contract_checks.json` for model checks, and `qa/visual_checks.json` for the chart review. `qa/verified_smoke_comparison.csv` compares the matching small cohorts. Earlier development attempts and logs are retained for traceability; interrupted attempts are not trained model deliverables.

The four original notebooks and the source release metadata match their recorded pre-work hashes. Start with the small default run, review the configuration, then choose pilot or full mode and your vessel groups as described above.

### Hardware/path update verification

The portable settings update passed 19 regression tests, an all-group full-mode metadata preflight without training, one Cargo neural CPU smoke run including checkpoint reload/calibration/evaluation, and a baseline notebook-worker run using custom job/result directories. Evidence is in `qa/portable_settings_checks.json` and the adjacent logs. GPU selection/probe behavior was tested with mocked hardware; actual GPU training remains unverified until run on the cluster. The four notebook files and their earlier saved outputs were not regenerated for this launcher update. Full training has not been started.
