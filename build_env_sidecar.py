"""Join Copernicus Marine environmental fields onto the prepared forecasting windows.

Writes one Parquet sidecar per release shard, row-aligned with that shard, under a separate
output tree. The release itself is frozen and checksummed and is never modified.

Design notes that matter:

* **Row-aligned, not key-joined.** Each sidecar mirrors its shard's relative path and row order,
  so the loader can apply the same row selection to both. `segment_id` and `origin_time_ns` are
  carried anyway and must be asserted on load; alignment that is merely assumed is alignment that
  silently rots.
* **Nearest wet cell, with the distance recorded.** Exact-cell lookup fails for most groups: a
  1.5 km ocean grid has no water inside the dock estate, and only 11% of Tanker window origins land
  on a wet current cell. Roughly 90% lie within one cell of water, so the sample is taken from the
  nearest wet cell of *that source* and the distance is recorded. Results must be stratified by it.
* **Per-source masks.** `zos` and the wave field share a mask (670 cells here); `uo`/`vo` are
  masked in 39 cells where sea level is defined. Every cell with currents has sea level, never the
  reverse. One shared validity flag would be wrong.
* **Bilinear where it is safe.** When all four surrounding cells are wet the sample is bilinear;
  otherwise it falls back to the nearest wet cell. `*_method` records which was used.
* **Directions are relative to the bow**, and stored as sine/cosine. Head, beam and following
  conditions act on a hull completely differently, and absolute bearing is close to useless without
  the vessel's orientation. Note VMDR is a *from* direction while currents are vectors; they are
  reconciled here rather than left as a trap.

Grid indexing uses rint((v - v0) / dv). An earlier analysis used np.searchsorted, which is off by
one against nearest-neighbour; that systematic shift pushed coastal vessels onto land and looked
like a physical finding rather than a bug. `_assert_grid` guards against a recurrence.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent
EARTH_M_PER_DEG = 111_320.0

# Blocks are separable so each source can be ablated independently.
SOURCES = {
    'cur': dict(file='cur_1p5km.nc', vars=['uo', 'vo']),
    'ssh': dict(file='ssh_1p5km.nc', vars=['zos']),
    'wav': dict(file='wav_1p5km.nc', vars=['VHM0', 'VTM02', 'VMDR', 'VHM0_SW1', 'VSDX', 'VSDY']),
    # ERA5, not UKV -- but the rationale is weaker than an earlier version of this comment claimed,
    # and is recorded here rather than in a commit message because it may yet invert.
    #
    # The claim was "the NW Shelf wave model is ERA5-forced, so ERA5 wind is consistent with the
    # wave field by construction". That is true of the NW Shelf wave *reanalysis*. The file this
    # module actually reads is `cmems_mod_nws_wav_anfc_1.5km_PT1H-i` -- the *analysis-forecast*
    # product, whose atmospheric forcing is NOT verified here. The catalogue entry for
    # NWSHELF_ANALYSISFORECAST_WAV_004_014 cites Bruciaferri et al. 2021 on the Met Office coupled
    # system, which suggests Unified Model forcing rather than ERA5. If that is so, the consistency
    # argument points at UKV and this comment currently has it backwards.
    #
    # VERIFY against the Product User Manual / QUID for NWSHELF_ANALYSISFORECAST_WAV_004_014 before
    # either choice is written into a paper. Until then ERA5 is used for reasons that hold
    # regardless: it is small, already downloaded, needs no CEDA licence, and lands on a regular
    # lat/lon grid. UKV is 1.5 km over the UK interior (not the 2 km DataHub markets) and sits on a
    # rotated-pole grid this module cannot index as written -- see TIDE_GAUGE_OPTION.md and the
    # plan's wind section for the access routes.
    'wnd': dict(file='wnd_era5.nc', vars=['u10', 'v10', 'msl']),
}


def _assert_grid(lat, lon):
    """Check the property nearest-cell indexing actually needs: that rint round-trips.

    CMEMS stores coordinates as float32, so consecutive differences jitter by ~1e-5 degrees. That
    is rounding, not an irregular grid, and testing diffs for equality rejects a perfectly usable
    axis. What matters is that rint((v - v0) / dv) returns each coordinate's own index for every
    coordinate on the axis — which is exactly the operation sample() performs.
    """
    out = []
    for name, v in (('latitude', lat), ('longitude', lon)):
        dv = (v[-1] - v[0]) / (len(v) - 1)          # mean step, robust to float32 jitter
        if dv <= 0:
            raise ValueError(f'{name} must ascend')
        idx = np.rint((v - v[0]) / dv).astype(int)
        if not np.array_equal(idx, np.arange(len(v))):
            bad = int(np.argmax(idx != np.arange(len(v))))
            raise ValueError(f'{name}: index round-trip failed at {bad}; grid is not uniform enough '
                             'for nearest-cell indexing')
        drift = np.abs((v - v[0]) / dv - np.arange(len(v))).max()
        if drift > 0.25:
            raise ValueError(f'{name}: cell-centre drift {drift:.2f} cells is too large')
        out.append(dv)
    return out[0], out[1]


def _open(path):
    """Open a source and normalise the conventions this module assumes.

    CMEMS files already satisfy them. ERA5 from the current CDS does not, in three ways, each of
    which is silent rather than loud: the time axis is named `valid_time`; latitude descends from
    north to south, which `_assert_grid` rejects; and ERA5T-merged files may carry a length-1
    `expver` dimension. A descending axis is the dangerous one — `rint((v - v0) / dv)` with a
    negative `dv` is rejected outright here rather than producing mirrored indices.
    """
    d = xr.open_dataset(path)
    if 'valid_time' in d.dims or 'valid_time' in d.coords:
        d = d.rename({'valid_time': 'time'})
    for axis in ('latitude', 'longitude'):
        if axis not in d.coords:
            raise ValueError(f'{path.name}: no {axis} coordinate')
        if d[axis].size > 1 and float(d[axis][-1]) < float(d[axis][0]):
            d = d.sortby(axis)
    for extra in ('expver', 'number'):
        if extra in d.dims:
            if d.sizes[extra] != 1:
                raise ValueError(f'{path.name}: {extra} has {d.sizes[extra]} members; pick one')
            d = d.isel({extra: 0}, drop=True)
    return d


class Field:
    """One NetCDF source, held in memory, with a nearest-wet-cell map."""

    def __init__(self, path, variables, max_cell_search=3):
        d = _open(path)
        missing = [v for v in variables if v not in d.data_vars]
        if missing:
            raise ValueError(f'{path.name}: missing variables {missing}')
        self.lat = d.latitude.values.astype('float64')
        self.lon = d.longitude.values.astype('float64')
        self.dla, self.dlo = _assert_grid(self.lat, self.lon)
        self.t0 = d.time.values[0]
        steps = np.diff(d.time.values)
        if len(np.unique(steps)) != 1:
            raise ValueError(f'{path.name}: time axis is not uniform')
        self.dt_ns = float(steps[0] / np.timedelta64(1, 'ns'))
        self.t0_ns = float(self.t0.astype('datetime64[ns]').astype('int64'))
        self.n_time = d.sizes['time']
        want = (self.n_time, len(self.lat), len(self.lon))
        self.data = {}
        for v in variables:
            arr = d[v].transpose('time', 'latitude', 'longitude').values.astype('float32')
            if arr.shape != want:
                raise ValueError(f'{path.name}: {v} has shape {arr.shape}, expected {want}')
            self.data[v] = arr
        first = self.data[variables[0]]
        self.wet = np.isfinite(first[0])
        if not self.wet.any():
            raise ValueError(f'{path.name}: no wet cells')
        self._build_nearest(max_cell_search)
        d.close()

    def _build_nearest(self, max_cell_search):
        """For every grid cell, the nearest wet cell and the distance to it in metres."""
        ii, jj = np.nonzero(self.wet)
        gi, gj = np.meshgrid(np.arange(len(self.lat)), np.arange(len(self.lon)), indexing='ij')
        # metres, so latitude and longitude spacing are comparable
        wy = ii * self.dla * EARTH_M_PER_DEG
        wx = jj * self.dlo * EARTH_M_PER_DEG * np.cos(np.radians(self.lat.mean()))
        qy = gi.ravel() * self.dla * EARTH_M_PER_DEG
        qx = gj.ravel() * self.dlo * EARTH_M_PER_DEG * np.cos(np.radians(self.lat.mean()))
        d2 = (qy[:, None] - wy[None, :]) ** 2 + (qx[:, None] - wx[None, :]) ** 2
        k = d2.argmin(1)
        self.near_i = ii[k].reshape(self.wet.shape)
        self.near_j = jj[k].reshape(self.wet.shape)
        self.near_m = np.sqrt(d2[np.arange(len(k)), k]).reshape(self.wet.shape).astype('float32')
        self.max_search_m = max_cell_search * max(self.dla, self.dlo) * EARTH_M_PER_DEG

    def sample(self, lat, lon, t_ns):
        """Return {var: values}, plus validity, distance and interpolation method per window."""
        n = len(lat)
        fi = (lat - self.lat[0]) / self.dla
        fj = (lon - self.lon[0]) / self.dlo
        i = np.clip(np.rint(fi).astype(int), 0, len(self.lat) - 1)
        j = np.clip(np.rint(fj).astype(int), 0, len(self.lon) - 1)
        in_box = (fi >= -0.5) & (fi <= len(self.lat) - 0.5) & (fj >= -0.5) & (fj <= len(self.lon) - 0.5)

        # time: linear between the two bracketing steps
        ft = (t_ns.astype('float64') - self.t0_ns) / self.dt_ns
        in_time = (ft >= 0) & (ft <= self.n_time - 1)
        t_lo = np.clip(np.floor(ft).astype(int), 0, self.n_time - 1)
        t_hi = np.clip(t_lo + 1, 0, self.n_time - 1)
        w_hi = np.clip(ft - t_lo, 0, 1).astype('float32')

        # bilinear only where all four surrounding cells are wet, else nearest wet cell
        i0 = np.clip(np.floor(fi).astype(int), 0, len(self.lat) - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, len(self.lon) - 2)
        quad_wet = (self.wet[i0, j0] & self.wet[i0 + 1, j0]
                    & self.wet[i0, j0 + 1] & self.wet[i0 + 1, j0 + 1])
        use_bilinear = quad_wet & in_box & in_time

        si, sj = self.near_i[i, j], self.near_j[i, j]
        dist = self.near_m[i, j].astype('float32')
        dist[self.wet[i, j]] = 0.0
        valid = in_box & in_time & (dist <= self.max_search_m)

        out = {}
        for v, arr in self.data.items():
            lo = arr[t_lo, si, sj]
            hi = arr[t_hi, si, sj]
            nearest = lo + (hi - lo) * w_hi
            if use_bilinear.any():
                a = fi - i0
                b = fj - j0
                def plane(tix):
                    return (arr[tix, i0, j0] * (1 - a) * (1 - b) + arr[tix, i0 + 1, j0] * a * (1 - b)
                            + arr[tix, i0, j0 + 1] * (1 - a) * b + arr[tix, i0 + 1, j0 + 1] * a * b)
                bil = plane(t_lo) + (plane(t_hi) - plane(t_lo)) * w_hi
                vals = np.where(use_bilinear, bil, nearest)
            else:
                vals = nearest
            out[v] = np.where(valid, vals, 0.0).astype('float32')
        method = np.where(use_bilinear, 'bilinear', np.where(valid, 'nearest_wet', 'none'))
        return out, valid, np.where(valid, dist, np.nan).astype('float32'), method


def _heading(X, mask):
    """Bow direction at the final input step, as a unit vector. NaN where heading is missing."""
    s, c = X[:, -1, 5], X[:, -1, 6]
    ok = mask[:, -1, 5] & mask[:, -1, 6] & (np.hypot(s, c) > 0.5)
    return np.where(ok, s, np.nan), np.where(ok, c, np.nan)


def _relative(to_sin, to_cos, h_sin, h_cos):
    """sin/cos of (field direction - heading), both given as unit vectors."""
    return to_sin * h_cos - to_cos * h_sin, to_cos * h_cos + to_sin * h_sin


def build_shard(npz_path, fields, transformer):
    with np.load(npz_path, allow_pickle=False) as z:
        xy = z['origin_xy_m']
        t_ns = z['origin_time_ns']
        seg = z['segment_id']
        X, Xm = z['X'], z['X_mask']
    lon, lat = transformer.transform(xy[:, 0], xy[:, 1])
    h_sin, h_cos = _heading(X, Xm)
    out = {'segment_id': seg.astype(str), 'origin_time_ns': t_ns}

    if 'cur' in fields:
        v, ok, dist, how = fields['cur'].sample(lat, lon, t_ns)
        speed = np.hypot(v['uo'], v['vo'])
        unit = np.where(speed > 1e-6, 1.0 / np.maximum(speed, 1e-6), 0.0)
        rs, rc = _relative(v['uo'] * unit, v['vo'] * unit, h_sin, h_cos)
        out.update(cur_east=v['uo'], cur_north=v['vo'],
                   cur_rel_sin=np.nan_to_num(rs), cur_rel_cos=np.nan_to_num(rc),
                   current_valid=ok, current_cell_distance_m=dist, current_method=how)

    if 'ssh' in fields:
        f = fields['ssh']
        v, ok, dist, how = f.sample(lat, lon, t_ns)
        ahead, _, _, _ = f.sample(lat, lon, t_ns + int(f.dt_ns))
        rate = (ahead['zos'] - v['zos']) / (f.dt_ns / 1e9)      # metres per second
        out.update(ssh=v['zos'], ssh_rate=rate.astype('float32'),
                   ssh_valid=ok, ssh_cell_distance_m=dist, ssh_method=how)

    if 'wav' in fields:
        v, ok, dist, how = fields['wav'].sample(lat, lon, t_ns)
        # VMDR is the direction waves come FROM; add 180 degrees to get the direction of travel,
        # so it is comparable with a heading and with the current vector.
        to = np.radians(v['VMDR'] + 180.0)
        rs, rc = _relative(np.sin(to), np.cos(to), h_sin, h_cos)
        out.update(wave_hs=v['VHM0'], wave_tm02=v['VTM02'],
                   wave_rel_sin=np.nan_to_num(rs), wave_rel_cos=np.nan_to_num(rc),
                   wave_hs_swell=v['VHM0_SW1'],
                   stokes_mag=np.hypot(v['VSDX'], v['VSDY']).astype('float32'),
                   wave_valid=ok, wave_cell_distance_m=dist, wave_method=how)

    if 'wnd' in fields:
        v, ok, _, how = fields['wnd'].sample(lat, lon, t_ns)
        # u10/v10 are velocity components, so they already point the way the air is going --
        # unlike VMDR above, which is a *from* direction and needed 180 degrees adding. Do not
        # "correct" this to match the wave block; the two conventions genuinely differ.
        speed = np.hypot(v['u10'], v['v10'])
        unit = np.where(speed > 1e-6, 1.0 / np.maximum(speed, 1e-6), 0.0)
        rs, rc = _relative(v['u10'] * unit, v['v10'] * unit, h_sin, h_cos)
        rs, rc = np.nan_to_num(rs), np.nan_to_num(rc)
        out.update(wind_speed=speed.astype('float32'),
                   wind_rel_sin=rs, wind_rel_cos=rc,
                   # beam component: what pushes a high-sided hull sideways, and the reason wind
                   # is carried at all once the wave field already encodes wind's effect on the sea
                   wind_cross=(speed * rs).astype('float32'),
                   mslp=v['msl'], wind_valid=ok, wind_method=how)

    out['heading_valid'] = np.isfinite(h_sin)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', default='../outputs/vessel_group_forecasting/data/full_5_60_20260918T072424Z')
    p.add_argument('--cache', default='../env_cache', help='Directory holding the downloaded NetCDF')
    p.add_argument('--out', default='../env_sidecar', help='Where the Parquet sidecars are written')
    p.add_argument('--sources', nargs='+', default=['cur', 'ssh', 'wav'], choices=list(SOURCES))
    p.add_argument('--groups', nargs='+', help='Restrict to these vessel groups')
    p.add_argument('--splits', nargs='+', help='Restrict to these roles')
    p.add_argument('--limit', type=int, help='Process at most this many shards (for a trial run)')
    a = p.parse_args()

    release, cache, out = Path(a.release).resolve(), Path(a.cache).resolve(), Path(a.out).resolve()
    if out.is_relative_to(release):
        raise ValueError('Sidecars must be written outside the frozen release')

    fields = {}
    for name in a.sources:
        spec = SOURCES[name]
        print(f'loading {name}: {spec["file"]}', flush=True)
        fields[name] = Field(cache / spec['file'], spec['vars'])
        f = fields[name]
        print(f'  {f.wet.sum()} wet cells, {f.n_time} steps, search radius {f.max_search_m:.0f} m')

    index = pd.read_csv(release / 'shard_index.csv')
    if a.groups:
        index = index[index['group'].isin(a.groups)]
    if a.splits:
        index = index[index['split'].isin(a.splits)]
    index = index.sort_values('file')
    if a.limit:
        index = index.head(a.limit)

    tr = Transformer.from_crs('EPSG:32630', 'EPSG:4326', always_xy=True)
    written = rows = 0
    for rec in index.to_dict('records'):
        frame = build_shard(release / rec['file'], fields, tr)
        if len(frame) != rec['windows']:
            raise ValueError(f'{rec["file"]}: {len(frame)} rows for {rec["windows"]} windows')
        target = out / (rec['file'].replace('.npz', '.parquet'))
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        written += 1
        rows += len(frame)
        if written % 100 == 0:
            print(f'  {written}/{len(index)} shards, {rows:,} windows', flush=True)

    # The two families carry different licences and different required wording.
    attribution = {}
    if {'cur', 'ssh', 'wav'} & set(a.sources):
        attribution['marine'] = 'Generated using E.U. Copernicus Marine Service Information'
    if 'wnd' in a.sources:
        attribution['atmosphere'] = 'Generated using Copernicus Climate Change Service information'

    manifest = dict(
        release=str(release), cache=str(cache), sources=a.sources, shards=written, windows=rows,
        datasets={'cur': 'cmems_mod_nws_phy-cur_anfc_1.5km-2D_PT1H-i',
                  'ssh': 'cmems_mod_nws_phy-ssh_anfc_1.5km-2D_PT1H-i',
                  'wav': 'cmems_mod_nws_wav_anfc_1.5km_PT1H-i',
                  'wnd': 'reanalysis-era5-single-levels'},
        attribution=attribution,
        wet_cells={k: int(v.wet.sum()) for k, v in fields.items()},
        search_radius_m={k: float(v.max_search_m) for k, v in fields.items()})
    (out / 'sidecar_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\n{written} shards, {rows:,} windows -> {out}')


if __name__ == '__main__':
    main()
