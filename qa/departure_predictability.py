"""Is departure predictable from the input history, or is it irreducible?

P2.0 design C puts a classifier in stage 1: does this vessel move at all? That is only worth building
if the answer is learnable. The K=2 mixture trained on Port_Service put identical weights on windows
that stayed and windows that departed (0.462 against 0.464), so an unsupervised gate extracted nothing.
A supervised classifier has a label to work with and may do better -- this measures whether it does.

Three feature sets, so the source of any signal is attributable:

  traj  trajectory history only -- what the forecasting model already sees
  +time hour of day and day of week, which the model does NOT see
  +tide sea level and its rate at the window origin, from the Copernicus field

The tidal block is the interesting one. Lock gates at Liverpool open near high water, so departures
are tide-gated by the port's physical constraints rather than by anything in the vessel's own track.
If tide predicts departure where trajectory history cannot, that is a direct link between the
environmental layer and the regime work, and it is a mechanism rather than a correlation.

AIS reporting cadence is included deliberately: Class A transmits every few minutes at a berth and
every few seconds under way, so a shortening observation_age is a plausible leading indicator of a
vessel about to sail. If the signal is there, that is where it should appear.
"""
import sys, json, hashlib, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from workflow import motion_regime, KNOTS_PER_MS
from build_env_sidecar import Field
from pyproj import Transformer

REL = Path(__file__).resolve().parents[2] / 'outputs/vessel_group_forecasting/data/full_5_60_20260918T072424Z'
CACHE = Path(__file__).resolve().parents[2] / 'env_cache'
DEPART_KN = 0.5          # same definition used throughout the regime measurements


def features(z, ssh_field, tr):
    X, M = z['X'], z['X_mask']
    sog = np.where(M[..., 2], X[..., 2], np.nan)
    age = np.where(M[..., 7], X[..., 7], np.nan)
    pos = X[..., :2]
    step = np.linalg.norm(np.diff(pos, axis=1), axis=-1)
    with np.errstate(invalid='ignore', all='ignore'):
        f = dict(
            sog_mean=np.nanmean(sog, 1), sog_max=np.nanmax(sog, 1), sog_std=np.nanstd(sog, 1),
            sog_last=sog[:, -1], sog_trend=np.nanmean(sog[:, -5:], 1) - np.nanmean(sog[:, :5], 1),
            path_len=step.sum(1), net_disp=np.linalg.norm(pos[:, -1] - pos[:, 0], axis=-1),
            # reporting cadence: shortens sharply as a vessel gets under way
            age_mean=np.nanmean(age, 1), age_last=age[:, -1],
            age_trend=np.nanmean(age[:, -5:], 1) - np.nanmean(age[:, :5], 1),
            hdg_std=np.nanstd(np.arctan2(np.where(M[..., 5], X[..., 5], np.nan),
                                         np.where(M[..., 6], X[..., 6], np.nan)), 1),
            valid_frac=M[..., 2].mean(1),
            east=z['origin_xy_m'][:, 0], north=z['origin_xy_m'][:, 1])
    t = pd.to_datetime(z['origin_time_ns'], utc=True)
    time_f = dict(hour=t.hour.to_numpy(), dow=t.dayofweek.to_numpy())
    lon, lat = tr.transform(z['origin_xy_m'][:, 0], z['origin_xy_m'][:, 1])
    v, ok, _, _ = ssh_field.sample(lat, lon, z['origin_time_ns'])
    ahead, _, _, _ = ssh_field.sample(lat, lon, z['origin_time_ns'] + int(ssh_field.dt_ns))
    tide_f = dict(ssh=np.where(ok, v['zos'], np.nan),
                  ssh_rate=np.where(ok, ahead['zos'] - v['zos'], np.nan))
    return f, time_f, tide_f


def build(group, split, n_shards, ssh_field, tr, idx, strict=False):
    """strict=True keeps only windows with max input SOG below the threshold at EVERY step.

    Without it the measurement is dominated by a near-tautology. 34% of Port_Service departures and
    53% of Tanker's already show motion somewhere inside the input window, against 2.4% and 0.1% of
    windows that stay -- so a classifier trivially separates them and scores an AUC near 1.0 while
    having predicted nothing useful. Those vessels have visibly begun to move; the forecasting model
    can see that too. The question design C actually depends on is the remainder: a vessel that is
    motionless throughout its history and sails anyway.
    """
    # Sampled across the split, never the first N. Shards are written per source bucket, so the
    # leading shards of a split are dominated by a few vessels: an early version of this script took
    # head(40) and drew a Port_Service training set whose every departure came from ONE vessel. The
    # resulting AUC of 0.97 measured nothing but that. Seeded choice, as select_plan does.
    pool = idx[(idx.group == group) & (idx.split == split)].sort_values('file').file.tolist()
    if n_shards and len(pool) > n_shards:
        # hashlib, not hash(): Python randomises string hashing per process, so hash() gave a
        # different shard sample on every run and the pooled AUC moved between 0.951 and 0.970
        # purely from the draw. select_plan uses sha256 for the same reason.
        seed = int(hashlib.sha256(f'{group}:{split}'.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        pool = [pool[i] for i in np.sort(rng.choice(len(pool), n_shards, replace=False))]
    files = pool
    base, timef, tidef, lab, ident = [], [], [], [], []
    for fn in files:
        with np.load(REL / fn, allow_pickle=False) as z:
            reg = motion_regime(dict(X=z['X'], X_mask=z['X_mask']), 0.5)
            keep = reg == 'not_under_way'
            if strict:
                sog = np.where(z['X_mask'][..., 2], z['X'][..., 2], np.nan)
                with np.errstate(all='ignore'):
                    mx = np.nan_to_num(np.nanmax(sog, 1), nan=0.0)
                keep = keep & (mx < DEPART_KN / KNOTS_PER_MS)
            if not keep.any():
                continue
            sub = {k: z[k][keep] for k in ('X', 'X_mask', 'origin_xy_m', 'origin_time_ns', 'y')}
            f, tf, df = features(sub, ssh_field, tr)
            base.append(pd.DataFrame(f)); timef.append(pd.DataFrame(tf)); tidef.append(pd.DataFrame(df))
            d60 = np.linalg.norm(sub['y'][:, -1, :], axis=-1)
            lab.append(d60 / 3600 / KNOTS_PER_MS >= DEPART_KN)
            ident.append(np.stack([z['MMSI'][keep].astype('int64'), sub['origin_time_ns']], 1))
    return (pd.concat(base, ignore_index=True), pd.concat(timef, ignore_index=True),
            pd.concat(tidef, ignore_index=True), np.concatenate(lab), np.concatenate(ident))


def count_events(ident, y, gap_ns=2 * 3600 * 10**9):
    """Independent departure events: same vessel, origins more than two hours apart.

    Windows stride by 60 seconds, so one departure yields a dozen or more near-identical labelled
    windows. Window counts therefore overstate the evidence by an order of magnitude and every
    interval computed from them is far too narrow. This is the number to quote.
    """
    if not y.any():
        return 0
    m, t = ident[y, 0], ident[y, 1]
    return sum(1 + int((np.diff(np.sort(t[m == v])) > gap_ns).sum()) for v in np.unique(m))


def main():
    idx = pd.read_csv(REL / 'shard_index.csv')
    tr = Transformer.from_crs('EPSG:32630', 'EPSG:4326', always_xy=True)
    ssh = Field(CACHE / 'ssh_1p5km.nc', ['zos'])
    rows = []
    for group, strict in [(g, s) for g in ('Port_Service', 'Tanker') for s in (False, True)]:
        trb, trt, trd, ytr, itr = build(group, 'train', 200, ssh, tr, idx, strict)
        teb, tet, ted, yte, ite = build(group, 'test', 100, ssh, tr, idx, strict)
        ev_tr, ev_te = count_events(itr, ytr), count_events(ite, yte)
        tag = 'strict' if strict else 'all'
        print(f'\n{group} [{tag}]: train {len(ytr):,} windows / {ev_tr} events ({100*ytr.mean():.2f}% depart)  '
              f'test {len(yte):,} windows / {ev_te} events ({100*yte.mean():.2f}% depart)  '
              f'vessels departing: train {len(np.unique(itr[ytr,0]))}, test {len(np.unique(ite[yte,0]))}')
        if yte.sum() < 30 or ytr.sum() < 30:
            print('  too few departures to evaluate'); continue
        sets = [('traj', [trb], [teb]), ('traj+time', [trb, trt], [teb, tet]),
                ('traj+time+tide', [trb, trt, trd], [teb, tet, ted])]
        for name, a, b in sets:
            Xtr = pd.concat(a, axis=1); Xte = pd.concat(b, axis=1)
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            rows.append(dict(group=group, pop=tag, features=name, test_events=ev_te,
                             base_rate=100*yte.mean(), auc=roc_auc_score(yte, p),
                             ap=average_precision_score(yte, p),
                             lift=average_precision_score(yte, p)/yte.mean()))
    print('\n=== departure predictability (AUC 0.5 = no signal; lift 1.0 = no better than base rate) ===')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:9.3f}'))


if __name__ == '__main__':
    main()
