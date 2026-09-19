"""Download ERA5 10 m wind and mean sea-level pressure for the Liverpool Bay footprint.

This is the one environmental source that does not come from Copernicus Marine. Waves, currents
and sea level are already in `env_cache/` from `copernicusmarine subset`; wind comes from the
Climate Data Store, which is a different service with its own account, its own credentials file
and its own licence wording.

Why ERA5 and not the Met Office UKV, which is four times finer: the NW Shelf wave model is
ERA5-forced, so ERA5 wind is consistent with the wave field *by construction*. UKV would give a
2 km atmosphere that did not drive those waves, on a rotated-pole grid `build_env_sidecar.py`
cannot index. Consistency is worth more here than resolution -- but see the resolution note below
before assuming this source will carry much weight.

**Resolution warning.** ERA5 is 0.25 deg, roughly 28 km. The vessel footprint is about 0.5 x 0.8
deg, so the whole study area spans two or three cells. This source is therefore close to
spatially constant across the fleet: it contributes a *temporal* signal (this hour was gusty,
that one calm) and almost no spatial one. Treat `wnd` as a low-expectation ablation block, and do
not be surprised when it moves nothing. The box below is deliberately padded beyond the footprint
so that the four bilinear corners exist rather than collapsing onto one cell.

Requests are made one month at a time and cached, so an interrupted run resumes instead of
restarting. The per-month files are concatenated at the end into the single `wnd_era5.nc` that
`build_env_sidecar.py` expects.
"""
from pathlib import Path
import argparse
import calendar
import sys

ROOT = Path(__file__).resolve().parent

# Padded one cell beyond the footprint (lat 53.286-53.764, lon -3.528 to -2.751) so that every
# window has four surrounding cells and can be sampled bilinearly rather than nearest-only.
AREA = [54.25, -4.25, 52.75, -2.25]          # N, W, S, E -- the order the CDS API wants

# The intersection of the marine spans: currents start 2024-08-04, waves 2024-08-06. Starting
# here keeps every source on one clock; see ENVIRONMENTAL_JOIN_PLAN.md, "Start dates differ".
START = (2024, 8)
END = (2026, 9)

VARIABLES = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT.parent / 'env_cache'))
    p.add_argument('--parts', default=None, help='Where per-month files live (default <out>/era5_parts)')
    p.add_argument('--dry-run', action='store_true', help='List what would be requested, fetch nothing')
    a = p.parse_args()

    out = Path(a.out)
    parts = Path(a.parts) if a.parts else out / 'era5_parts'
    parts.mkdir(parents=True, exist_ok=True)

    todo = list(months(START, END))
    print(f'{len(todo)} months, {AREA[0]}N {AREA[2]}N / {AREA[1]}E {AREA[3]}E -> {parts}')
    if a.dry_run:
        for y, m in todo:
            f = parts / f'era5_{y}{m:02d}.nc'
            print(f'  {y}-{m:02d}  {"cached" if f.exists() else "would fetch"}')
        return

    try:
        import cdsapi
    except ModuleNotFoundError:
        sys.exit('cdsapi is not installed. See the header of this file, or: pip install cdsapi')

    client = cdsapi.Client()
    for y, m in todo:
        f = parts / f'era5_{y}{m:02d}.nc'
        if f.exists() and f.stat().st_size > 0:
            print(f'  {y}-{m:02d}  cached')
            continue
        ndays = calendar.monthrange(y, m)[1]
        print(f'  {y}-{m:02d}  requesting...', flush=True)
        client.retrieve('reanalysis-era5-single-levels', {
            'product_type': ['reanalysis'],
            'variable': VARIABLES,
            'year': [str(y)],
            'month': [f'{m:02d}'],
            'day': [f'{d:02d}' for d in range(1, ndays + 1)],
            'time': [f'{h:02d}:00' for h in range(24)],
            'area': AREA,
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        }, str(f))

    # Concatenate. Kept separate from the fetch loop so a re-run after a partial download does the
    # merge without re-requesting anything.
    import xarray as xr
    files = sorted(parts.glob('era5_*.nc'))
    if not files:
        sys.exit('no per-month files to merge')
    print(f'\nmerging {len(files)} files...')
    ds = xr.open_mfdataset(files, combine='by_coords')
    # Normalisation is left to build_env_sidecar._open, which already handles ERA5's valid_time
    # axis, descending latitude and length-1 expver. Writing them through unchanged keeps one
    # place responsible for those conventions instead of two that can disagree.
    target = out / 'wnd_era5.nc'
    ds.to_netcdf(target)
    span = f'{str(ds.indexes[list(ds.dims)[0] if "time" in ds.dims else "valid_time"][0])[:10]}'
    print(f'{target}  {target.stat().st_size / 1e6:.0f} MB  from {span}  vars {list(ds.data_vars)}')


if __name__ == '__main__':
    main()
