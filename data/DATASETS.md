# Data folder guide

## `raw/`  — what `src/preprocessing.py` actually reads
This is the **only** folder the pipeline (`train.py`, `evaluate.py`) looks at.
It expects the real **NASA PCoE Li-Ion Battery Aging Data Set** — one `.mat`
file per cell (`B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`, ...), each
containing a top-level struct with a `cycle` array whose `discharge` entries
have `data.Voltage_measured`, `data.Current_measured`,
`data.Temperature_measured`, `data.Time`, `data.Capacity`.

**It's currently empty.** I couldn't download it into this environment — my
sandbox's outbound network is restricted to an allowlist that
`phm-datasets.s3.amazonaws.com` isn't on (confirmed: the request came back
`403 host_not_allowed`). Until real files are here, every entry point
auto-falls-back to the synthetic generator (`generate_synthetic_nasa_dataset`),
which is fine for testing the pipeline but not for genuine results.

**To fix**, download and unzip, then drop the `.mat` files straight into
`data/raw/`:

- Direct link: <https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip>
- Source page: NASA PCoE Data Set Repository, entry **"5. Batteries"**
  (B. Saha and K. Goebel, 2007)

```bash
curl -L -o battery.zip "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
unzip battery.zip -d /tmp/nasa_battery
find /tmp/nasa_battery -name "*.mat" -exec cp {} data/raw/ \;
```

## `external/lot_reliability/`  — the two datasets you uploaded
This is **not** used by the current pipeline — it's kept here for reference /
future work, not wired into `preprocessing.py`.

- `Qualified lots.xlsx`, `Subsequent lots for ongoing reliability testing.xlsx`
  — each row is a cycle index, each of the 6 columns a device, values are a
  normalized degradation ratio (starts at 1.0, decays slowly) — no
  voltage/current/temperature channels.
- `Dataset2.mat` — a single `Lot1` variable: 23 cells × (cycle, ratio) pairs.
  Per `Notes.txt`, cells 1–14 are the qualified lot and 15–23 are from
  subsequent lots.

These look like electronic-component (not Li-ion cell) accelerated-life /
lot-qualification degradation data — a single scalar health ratio per cycle
per device, rather than per-cycle voltage/current/temperature telemetry for a
battery. I left them as-is rather than reshaping them to look like NASA
PCoE battery cycles, since that would fabricate channels that aren't in the
source data. `parse_nasa_mat_file` will skip `Dataset2.mat` harmlessly if it's
ever left in `raw/` (no `.cycle` field), so keeping it in `external/` avoids
any ambiguity about what's real telemetry vs. synthetic fallback.

If you tell me more about where Dataset1/Dataset2 came from (or want me to
build a separate loader for this scalar degradation-ratio format), I can
wire it in as its own dataset class rather than forcing it through the
battery preprocessing path.
