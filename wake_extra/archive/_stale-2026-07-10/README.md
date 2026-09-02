# Staged-aside wake caches (2026-07-10)

These are the `northsea/` and `europe/` caches as they stood before the
harmonised rerun for the Applied Energy wake paper.

## Why they were moved

`wake_helpers` keys its caches on `(clusters, technology, threshold)` plus, for
profiles, `(bias, correction_factor)`. The key does **not** include the cutout,
the snapshot range, the turbine, `capacity_per_sqkm`, or any of the land-use
exclusion settings. `cluster_network` compounds this: it writes the split-region
geojsons only `if not cache_path.is_file()`.

So a run under `config/pypsa-wake/` would have silently reused availability
matrices, solar profiles and offshore region geometries built in February under
a different configuration. That defeats the whole point of the harmonised
rerun, and it is the same mechanism that produced the inconsistent
`figures/wakes/*.csv`.

## Restoring

Nothing here is tracked by git. To go back:

```bash
cd wake_extra
rm -rf northsea europe
mv _stale-2026-07-10/northsea _stale-2026-07-10/europe .
```

Once the rerun has produced fresh caches and the results are validated, this
directory (≈3 GB) can be deleted.

## Not moved

`wake_extra/new_more_fit/` — the tiered-density breakpoint fit. It is an output
of `fit_new_more_breakpoints.py`, not a cache, and it is tracked in git.
