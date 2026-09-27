# LA Encampment Risk — Interactive Digital Twin (SNN-2)

Zoomable map of LA's hex grid. Click a hexagon to load its real data into
sliders across 5 categories; the map, risk metrics, and headline
interaction update live as you adjust them. Presentation-friendly layout:
map + risk + top interaction always visible up top, everything else
(sliders / contribution charts / optimizer) organized into tabs so a
presenter clicks between sections instead of scrolling through one long page.

**Uses SNN-2 exclusively.**

## Run it

```bash
pip install -r requirements.txt
streamlit run scripts/app.py
```

## Architecture fidelity notes

This build was reconciled against the reference notebook pipeline
(`Latest_Tent_analysis.ipynb`, cells 7-69 for quarter, 50-69 for half/
three_fourth-equivalent) for both data and architectural fidelity:

- **Labels come from the outputLayer join, not the enriched export's own
  `Join_Count`.** Each resolution's enriched export (`*_variables_and_
  tent(s)_geo_enriched.csv`) carries a stale `Join_Count` from a
  different/earlier spatial join. `prep.py` now merges each export with
  its authoritative `outputLayer_0_{7,9,10}.csv` join table on
  `OBJECTID` and takes `Join_Count` from *that* table — this reproduces
  the paper's tent-present rates (25.21% / 38.59% / 46.38%) exactly.
  (An earlier version of this pipeline grouped the enriched export by
  `GRID_ID` and used its own `Join_Count`, which gave the wrong, higher
  rates of 32.9% / 47.2% / 54.7% — see "Resolved gaps" below.)
- **`GRID_ID` is not the same address scheme in both files** for the
  same physical hex (`OBJECTID` is the reliable cross-file key; the two
  files' `GRID_ID` values agree on only ~0.14% of hexes). `prep.py` keeps
  the enriched export's own `GRID_ID` (paired with that row's own
  lat/lon) for centroid reconstruction, while still using the
  outputLayer's `Join_Count` for the label.
- **Feature lists restored verbatim**, including the 0.25mi resolution's
  12 extra "raw Esri" columns. These are **not** duplicates of the
  similarly-named renamed columns — `outputLayer_0_7.csv` carries its
  own, genuinely different second Esri/ACS enrichment pass (a different
  reference vintage), and `prep.py` now merges those columns in under
  their real long names with their own true per-hex values, rather than
  faking them as copies (see "Resolved gaps" below).
- **True 2D joint-RBF interaction kernel** (`BivariateRBFSubNet`, 6x6=36
  basis functions per pair, taking both raw feature values as separate
  inputs), matching the reference's `SubNet2` — not a collapsed
  single-column product approximation.
- **Main-effect layer is frozen before interaction training** — SNN-2
  fits interaction subnets only on the residual left after SNN-1, it
  does not retrain main effects jointly with interactions.
- **Pair selection** ranks by fitting a shallow RF directly on the
  pairwise product columns against hard labels (matching
  `select_top_pairs`), not combined with base features or soft labels.
- **Plain random 5-fold CV**, not spatially blocked — the reference
  code doesn't spatially block, and matches the paper's published
  numbers without it; an earlier version of this pipeline added spatial
  blocking based on a methodological read of the paper's prose, which
  turned out not to reflect what the actual reference code does.

### Current fidelity

Retrained on the corrected data/labels, matched against the reference
notebook's own printed output for each resolution:

| Resolution   | Hexes | Tent-present | RF teacher AUC (ours / ref) | SNN-2 AUC (ours / ref) |
|--------------|------:|-------------:|:----------------------------|:------------------------|
| quarter      | 1448  | 25.21%       | 0.7423 / 0.7423 (exact)     | 0.7788 / 0.776          |
| half         |  767  | 38.59%       | 0.6054 / 0.6054 (exact)     | 0.7152 / 0.714          |
| three_fourth |  539  | 46.38%       | 0.6143 / 0.6143 (exact)     | 0.7207 / 0.723          |

Top selected interaction pair also matches the reference exactly for
half (`pct_public_assistance × osm_amenity_count`) and three_fourth
(`business_sites_count × bare_ground_proxy_pct`); quarter's top pair
(`2023 Pop w/Income Below Poverty Level (ACS 5-Yr): Percent ×
police_station_count`) was not independently cross-checked against a
notebook printout (cell 49's own top-pair line was truncated in the
available output) but its RF/SNN-2 AUCs match to 4 decimal places /
within 0.003, so the underlying data and training are confirmed correct.

### Resolved gaps (this rebuild)

Two independent bugs were found and fixed against a fresh, complete copy
of the reference notebook and fresh raw exports:

1. **Wrong label source** (all three resolutions) — described above.
   Fixed in `prep.py`'s `load_and_merge_hexes()`.
2. **Faked quarter-resolution "raw Esri" columns** — a previous version
   of `train_snn2.py` carried a `QUARTER_ALIASES` dict that overwrote
   the 12 raw-Esri-named columns with copies of the similarly-named
   *renamed* columns, on the assumption they were exact duplicates. They
   are not: the real columns come from `outputLayer_0_7.csv`'s own,
   independent second enrichment pass and disagree with the renamed
   columns on ~99% of rows. Faking them destroyed real signal the
   reference RF was trained on, which is why quarter's RF teacher AUC
   was stuck at 0.61 (vs the reference's 0.74) even after the label fix
   above. `prep.py` now merges these columns in directly under their
   real long names with their genuine values, and `QUARTER_ALIASES` has
   been removed from `train_snn2.py` entirely.

## What's in here

```
data/
  hex_lookup_{quarter,half,three_fourth}.csv   # one row per hex: all raw
    enrichment columns (+ quarter's 12 aliased Esri columns) + reconstructed
    centroid + SNN-2 baseline_risk + top_interaction_a/b/val for the map

models/
  snn2_{res}.json                  # main_nets + pair_nets + both scalers
  snn2_feature_summary_{res}.csv   # main + interaction rows, delta_sj, slider bounds
  snn2_metrics_{res}.txt

scripts/
  prep.py              # raw CSV -> one row per hex + centroid reconstruction
  snn_core.py           # RBFSubNet (main effects) + BivariateRBFSubNet
                         # (interactions) + backfitting training loops
  train_snn2.py         # full SNN-2 training pipeline, fidelity-matched
  snn2_inference.py     # pure-numpy inference: risk + main/interaction
                         # contributions + batch_top_interactions()
  optimizer2.py          # single-hex interaction-aware greedy optimizer
  scenarios.py            # population-level optimizer (per the project's
                           # "Simple SNN-2 What-If Digital Twin" methodology
                           # doc's frozen/non-frozen variable-role table) --
                           # runs the greedy optimizer on every hex in the
                           # highest-risk N%, then aggregates the results
  hexgeom.py             # hexagon polygon geometry + tooltip properties
  labels2.py             # human-readable names/groups for all features
                          # incl. the quarter-only Esri alias columns
  app.py                 # the Streamlit app (map/headline always visible,
                          # Scenarios/Sliders/Contributions/Optimizer in tabs)
```

## Two ways to explore "what reduces risk," side by side

Both now respect the same frozen-variable rule from the project's
variable-role table: socioeconomic, demographic, crime, hazard, roadway,
hydrology, and land-cover variables are never candidates for change, in
either the Scenarios tab or any optimizer mode. Only Facilities/Services
variables are ever touched.

- **Scenarios tab**: runs the greedy optimizer (increase-only) independently
  on each of the highest-risk N% hexes, then aggregates: mean risk before/
  after across that population, a ΔRisk map, and which variables got
  recommended most often across the targeted hexes.
- **Optimize This Hex tab**: single-hex, three modes over the SAME
  non-frozen candidate set, differing only in direction allowed:
  - *Actionable*: increase-only (safe to present as real interventions)
  - *Exploratory*: increase or decrease (a decrease, e.g. fewer libraries,
    isn't a realistic intervention either — shows the fitted surface's
    full behavior within the non-frozen set, not a real-world action list)
  - *Combined*: same as Exploratory, but every step tagged ✅ Actionable
    (increase) or ⚠️ Correlational (decrease), with a separate re-scored
    "actionable-only" risk

## Known limitations

1. Hex centroids/boundaries are reconstructed/approximate, not the
   original ArcGIS tessellation geometry.
2. Quarter's top selected interaction pair wasn't independently
   cross-checked against a notebook printout (see "Current fidelity"
   above) — everything else has been verified against the reference
   notebook's own printed output.
