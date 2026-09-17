"""Portable command-line companion. Run --help for hardware and path options."""
import argparse
import json
from workflow import config, run_experiment, plot_run, release_info, select_plan, GROUPS, ROLES


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=['baselines', 'bilstm_attention', 'bilstm_only', 'transformer'])
    p.add_argument('--mode', choices=['smoke', 'pilot', 'full'], default='smoke')
    p.add_argument('--groups', nargs='+', choices=(*GROUPS, 'all'), default=['Cargo'])
    p.add_argument('--device', choices=['cpu', 'gpu', 'auto'], help='Override AIS_DEVICE in .env')
    p.add_argument('--env-file', help='Alternative .env file (default: beside this script)')
    p.add_argument('--data-dir', help='Prepared release folder containing manifest.json and shard_index.csv')
    p.add_argument('--runs-dir', help='Saved models and evaluation output directory')
    p.add_argument('--jobs-dir', help='Notebook worker request/log directory')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--beta-nll', type=float, help='beta-NLL weighting for every group, overriding the zero-inflated per-group defaults (0 reproduces plain Gaussian NLL)')
    p.add_argument('--monitor', choices=['val_ade_m', 'val_loss'], help='Quantity used to select checkpoints, reduce the learning rate and stop early')
    p.add_argument('--check-config', action='store_true', help='Validate release metadata, display cohort counts and check hardware; do not train')
    return p


def main():
    p = parser()
    a = p.parse_args()
    try:
        cfg = config(a.mode, a.groups, device=a.device, release=a.data_dir,
                     runs_dir=a.runs_dir, jobs_dir=a.jobs_dir, env_file=a.env_file,
                     batch_size=a.batch_size)
        if a.epochs is not None:
            if a.epochs < 1:
                raise ValueError('epochs must be a positive integer')
            cfg['epochs'] = a.epochs
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    if a.deterministic:
        cfg['probabilistic'] = False
    if a.beta_nll is not None:
        if a.beta_nll < 0:
            p.error('beta-nll must not be negative')
        # An explicit value applies to every group, replacing the per-group defaults.
        cfg['beta_nll'] = a.beta_nll
        cfg['beta_nll_by_group'] = {}
    if a.monitor is not None:
        cfg['monitor'] = a.monitor
    if a.check_config:
        _, index, _ = release_info(cfg)
        if a.model != 'baselines':
            from runtime_control import configure_runtime
            configure_runtime(cfg)
        print(json.dumps(cfg, indent=2))
        for group in cfg['groups']:
            counts = {role: sum(x['n'] for x in select_plan(index, group, role, cfg)) for role in ROLES}
            print(group, counts)
        print('Configuration check complete. No training or full shard checksum scan performed.')
        return
    run = run_experiment(a.model, cfg)
    plot_run(run)
    print(run)


if __name__ == '__main__':
    main()
