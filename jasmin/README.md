# Upload and submit the AIS training workflow on JASMIN

These scripts call the same shared training code as notebooks 01–04. They add queue wrappers only; the original notebooks, model code, dataset and running local jobs are unchanged. They have been checked locally for shell syntax and routing, but have not been submitted or executed on JASMIN.

## 1. Upload code from your Mac

Upload `training_5_60_v1`, including this `jasmin` directory and all top-level Python files. You may omit old results and caches. The following command copies the folder contents into a new JASMIN code folder without deleting remote files or replacing a remote `.env`:

```bash
rsync -avP --exclude='runs/' --exclude='jobs/' --exclude='qa/' \
  --exclude='__pycache__/' --exclude='.cache/' --exclude='.ipynb_checkpoints/' \
  --exclude='.DS_Store' --exclude='.env' --exclude='config.sh' \
  /Users/shuvo/anaconda_projects/ais_liverpool/training_5_60_v1/ \
  shakib@xfer-vm-01.jasmin.ac.uk:/home/users/shakib/projects/training_5_60_v1/
```

The dataset is separate. Extract your uploaded ZIP and locate the directory whose immediate children include `groups`, `manifest.json` and `shard_index.csv`. Do not point at the ZIP itself or guess its nesting.

## 2. Get onto a submission server and check access

JASMIN documents job submission from scientific analysis servers. From your current JASMIN login session:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
ssh -A username@login.jasmin.ac.uk
```

Once logged into JASMIN, run:
```bash
ssh shakib@sci-vm-01.jasmin.ac.uk
cd /home/users/shakib/projects/training_5_60_v1
module load jaspy/3.11/v20240815
python --version
```

```bash
ssh sci-vm-01
cd /home/users/shakib/projects/training_5_60_v1
useraccounts
```

If onward SSH authentication fails, reconnect from your Mac using your registered key with agent forwarding, as described in JASMIN's login guide. GPU jobs require the `orchid` access role; a login account alone does not grant it. CPU baseline jobs need an accounting group shown by `useraccounts`.

## 3. Configure the Linux environment and paths once

Use a Linux Python 3.11 environment installed on storage visible to compute nodes; do not copy your Mac Python environment. If you already have a compatible environment, use it. Otherwise, after making Python 3.11 available on JASMIN, an example isolated setup is:

```bash
python3.11 -m venv /home/users/shakib/projects/ais_venv
/home/users/shakib/projects/ais_venv/bin/python -m pip install -r jasmin/requirements-gpu.txt
/home/users/shakib/projects/ais_venv/bin/python -m pip check
```

This is a proposed installation, not a verified ORCHID environment. If Python 3.11 is unavailable, use JASMIN's environment guidance or provide the available environment/version for adaptation. For CPU-only work the original `requirements.txt` suffices. The GPU requirements preserve the existing package pins and add matching TensorFlow pip NVIDIA runtime dependencies; a working NVIDIA driver is still required. Do not blindly load a different system CUDA toolkit. The queued smoke test checks GPU availability and runs an actual neural training epoch.

```bash
cp -n jasmin/config.sh.example jasmin/config.sh
nano jasmin/config.sh
```

Fill in `AIS_PYTHON`, the actual extracted `AIS_DATASET_DIR`, and persistent writable result/log paths. Add environment activation/module setup to this shell file if your chosen environment requires it. This configuration exports the same `AIS_*` variables used by training; explicit queue device arguments override `.env`. The Mac `.env` is unnecessary here. Keep enough quota for packages, the ZIP, extracted data and outputs; storing code under `projects` does not create a storage allocation.

## 4. Run a short GPU check first

Run from the code folder on the sci server, after access, environment and paths are ready:

```bash
sbatch --array=0 --time=00:30:00 jasmin/train_gpu.sbatch bilstm_attention smoke
squeue -u shakib
```
sbatch jasmin/train_gpu.sbatch bilstm_attention full

Check the resulting `slurm-ais-gpu-JOBID_0.out` and `.err` files. A successful job prints a selected GPU, completes a Cargo epoch, and saves a completed run directory. The memory request (32 GB) and CPU request (4) are initial settings, not measured ORCHID resource requirements. The time request limits runtime; it does not estimate queue wait time.

## 5. Submit full models

After the GPU smoke check succeeds:

```bash
# Notebook 02
sbatch jasmin/train_gpu.sbatch bilstm_attention full
# Notebook 03
sbatch jasmin/train_gpu.sbatch bilstm_only full
# Notebook 04
sbatch jasmin/train_gpu.sbatch transformer full
# Notebook 01: replace YOUR_ACCOUNT with the correct entry from useraccounts.
sbatch --account=YOUR_ACCOUNT jasmin/train_cpu.sbatch baselines full
```

Each command submits a seven-task array: 0 Cargo, 1 Tanker, 2 Passenger, 3 Port_Service, 4 Research_Offshore, 5 Fishing, 6 Unknown. The default `0-6%1` runs one group at a time per array, with a new allocation and up to 24 hours for each group. Each task writes a separate timestamped run folder under `AIS_RUNS_DIR`. This preserves specialist training and data selection, but results are now spread across seven run folders rather than one combined folder. Do not pass `all` to these wrappers; the array already covers all groups.

Use `--array=0` to submit just Cargo, or `--array=0-1%1` for Cargo and Tanker. Use `pilot` in place of `full` for the corresponding pilot. Submit model families individually if you want to review resource usage first.

## Time limits and completion

The standard ORCHID QoS currently permits 24 hours; an on-request 48-hour QoS also exists. Individual-group submission avoids consuming the whole allocation on early groups, but does not guarantee that large groups will finish in 24 hours. The workflow has no automatic resume support. A timed-out task may retain a partial checkpoint, and its run metadata can still say `running` after a hard termination. Check Slurm status as well as the run's `complete` status; do not treat a partial checkpoint as a fully calibrated/evaluated model. No automatic requeue is requested. Measure the GPU run time before relying on a full 100-epoch job.

```bash
squeue -u shakib
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

Successful jobs save `best_model.keras`, the scaler, calibration and evaluation outputs inside their group directory. Baselines save evaluation outputs rather than neural weights. Queue output files are separate from the notebook `jobs/` logs.

## Sources checked 14 September 2026

- [ORCHID access, submission directives and limits](https://help.jasmin.ac.uk/docs/batch-computing/orchid-gpu-cluster/)
- [Submission from sci servers](https://help.jasmin.ac.uk/docs/batch-computing/how-to-submit-a-job/)
- [CPU accounts and queues](https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/)
- [Scientific analysis servers](https://help.jasmin.ac.uk/docs/interactive-computing/sci-servers/)
- [Python environments](https://help.jasmin.ac.uk/docs/software-on-jasmin/python-virtual-environments/)
- [TensorFlow installation](https://www.tensorflow.org/install/pip)
