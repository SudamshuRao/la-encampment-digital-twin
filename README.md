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
  different/earlier spatial join. `prep.py` merges each export with its
  authoritative `outputLayer_0_{7,9,10}.csv` join table and takes
  `Join_Count` from *that* table — this reproduces the paper's
  tent-present rates (25.21% / 38.59% / 46.38%) exactly.
- **The cross-file join key is `GRID_ID`, not `OBJECTID` — this was
  found and fixed after the numbers above were first reported.** An
  earlier version of this pipeline joined the enriched export to the
  outputLayer table on `OBJECTID`, reasoning that `OBJECTID` was the
  reliable cross-file key and `GRID_ID` was not (the enriched export
  has 2-4 rows per physical hex, one per street-view-image heading, so
  a naive `GRID_ID` merge fans out). That reasoning had it backwards on
  both counts, confirmed via direct geometry-equality and
  `sjoin_nearest` spatial tests against fresh raw exports: `OBJECTID`
  is each file's own independent row-counter with near-zero (~0.05-0.2%)
  correspondence to the same physical hex across files, while `GRID_ID`
  (e.g. `"AS-43"`) *is* the stable, ~100%-matching spatial address once
  deduplicated within a file (every demographic/crime/facility/hazard/
  climate feature column is byte-identical across a `GRID_ID`'s
  duplicate rows — only per-photo metadata differs). The OBJECTID join
  produced the *correct label rates* (they come straight from the
  outputLayer side, which is unaffected) paired with *essentially
  random features* — a data-correctness bug, not a rate bug, so it
  didn't show up in any of the headline numbers below; only the
  underlying feature-to-label pairing was wrong. `prep.py` now
  deduplicates the enriched export by `GRID_ID` and joins on `GRID_ID`
  on both sides. See "Resolved gaps" below for the retrained AUC deltas
  this produced.
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

An earlier version of this table reported RF/SNN-2 AUCs that matched the
reference notebook's own printed output almost exactly. That match was
real, but it was matching a bug: direct inspection of the reference
notebook (`Latest_Tent_analysis.ipynb`) confirmed it performs the exact
same `OBJECTID`-based cross-file join described above — so its own
reported numbers were trained on the same randomly-mismatched
feature/label pairing ours were. Reproducing them exactly was evidence
we'd faithfully copied the reference pipeline, not evidence the pipeline
was correct.

The numbers below are retrained on the `GRID_ID`-joined, corrected data
and **intentionally no longer match the reference notebook** — tent-
present rates are unchanged (the label side was never wrong), but every
AUC is now higher, most dramatically for half/three_fourth, which had
the most severely scrambled features under the old join:

| Resolution   | Hexes | Tent-present | RF teacher AUC (corrected / old-join reference) | SNN-2 AUC (corrected / old-join reference) |
|--------------|------:|-------------:|:-------------------------------------------------|:---------------------------------------------|
| quarter      | 1448  | 25.21%       | 0.7659 / 0.7423                                   | 0.7875 / 0.7788                               |
| half         |  767  | 38.59%       | 0.7882 / 0.6054                                   | 0.8064 / 0.7152                               |
| three_fourth |  539  | 46.38%       | 0.8244 / 0.6143                                   | 0.8351 / 0.7207                               |

The top selected interaction pair also changed for half and three_fourth
now that the correct features are attached to each label (previously
`pct_public_assistance × osm_amenity_count` and `business_sites_count ×
bare_ground_proxy_pct`; now `affordable_housing_count ×
business_sites_count` for both) — see `models/snn2_feature_summary_
{res}.csv` for the full ranked list per resolution.

### Resolved gaps (this rebuild)

Three independent bugs were found and fixed against a fresh, complete
copy of the reference notebook and fresh raw exports:

1. **Wrong label source** (all three resolutions) — described above.
   Fixed in `prep.py`'s `load_and_merge_hexes()`.
2. **Faked quarter-resolution "raw Esri" columns** — a previous version
   of `train_snn2.py` carried a `QUARTER_ALIASES` dict that overwrote
   the 12 raw-Esri-named columns with copies of the similarly-named
   *renamed* columns, on the assumption they were exact duplicates. They
   are not: the real columns come from `outputLayer_0_7.csv`'s own,
   independent second enrichment pass and disagree with the renamed
   columns on ~99% of rows. Faking them destroyed real signal the
   reference RF was trained on. `prep.py` now merges these columns in
   directly under their real long names with their genuine values, and
   `QUARTER_ALIASES` has been removed from `train_snn2.py` entirely.
3. **Wrong cross-file join key** (`OBJECTID` instead of `GRID_ID`, all
   three resolutions, including the reference notebook itself) —
   described above under "Architecture fidelity notes." This is the
   fix behind the AUC jumps in "Current fidelity": every hex's label
   was already correct, but was frequently paired with an unrelated
   hex's demographic/crime/facility/hazard features. Fixed in `prep.py`
   (dedup by `GRID_ID`, join on `GRID_ID`) and in `hexgeom.py` / the map
   geometry build (see "Real hex geometry" below — the "roughly half
   the hexes have no data" gaps reported in an earlier version of this
   README turned out to be the same bug, not real missing coverage).

## What's in here

```
data/
  hex_lookup_{quarter,half,three_fourth}.csv   # one row per hex WITH model
    data: all raw enrichment columns (+ quarter's 12 aliased Esri columns)
    + reconstructed centroid + SNN-2 baseline_risk + top_interaction_a/b/val
  hex_geom_{quarter,half,three_fourth}.geojson  # real ArcGIS hex polygons,
    EVERY hex ArcGIS exported (~2x hex_lookup's row count -- see "Real hex
    geometry" below), each tagged OBJECTID / GRID_ID / has_data

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
  hexgeom.py             # loads real ArcGIS hex geometry (hex_geom_*.geojson)
                          # and merges it with live model data for the map
  labels2.py             # human-readable names/groups for all features
                          # incl. the quarter-only Esri alias columns
  app.py                 # the Streamlit app (map/headline always visible,
                          # Scenarios/Sliders/Contributions/Optimizer in tabs)
```

## Real hex geometry (replacing the reconstructed approximation)

The map renders each resolution's **actual ArcGIS-exported hex polygons**
(`data/hex_geom_{res}.geojson`), not a reconstructed regular-hexagon
approximation.

- **Every physical hex now has real model data — there are no true
  gaps.** An earlier version of this README reported that roughly half
  of each resolution's raw hexes (1430/2878 quarter, 1544/2311 half,
  1611/2150 three_fourth) had no matching row and carried no
  `baseline_risk`/`tent_present`, and rendered those as flat gray
  "no data" filler hexes so the map read as one contiguous surface. That
  turned out to be the OBJECTID-join bug described above, not real
  missing coverage: the raw counts above are the *un-deduplicated* row
  counts (2-4 street-view-heading rows per physical hex), and once
  deduplicated by `GRID_ID` every one of the study grid's 1448/767/539
  physical hexes has a real, correctly-paired row in `hex_lookup_
  {res}.csv` and a real polygon in `hex_geom_{res}.geojson` — confirmed
  0 missing on rebuild for all three resolutions. The "no data" render
  path (`has_data`) is kept in the schema/style function defensively but
  should never actually fire on this data.
- **GRID_ID and OBJECTID are both 1:1 with a real hex in these files.**
  The enriched export's raw duplicate-`GRID_ID` rows (one per street-
  view heading) are collapsed to a single row per physical hex during
  prep, before either `hex_lookup_{res}.csv` or `hex_geom_{res}.geojson`
  is written, so there's no remaining cross-file ambiguity between the
  two identifiers in this app.

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

1. `centroid_lat`/`centroid_lon` (used for a couple of secondary display
   purposes, not the map's polygons, which are now the real ArcGIS
   geometry) are still an affine-regression approximation from each
   hex's `GRID_ID` address, not surveyed centroids.
2. This rebuild's numbers **intentionally diverge** from the reference
   notebook's own printed output (see "Current fidelity" above) — the
   reference notebook shares the same `OBJECTID`-join bug this rebuild
   fixes, so no longer matching it is the point, not a regression. No
   independent, bug-free ground truth outside this project's own data
   was available to validate against beyond the internal checks
   described above (GRID_ID cross-file correspondence, feature-column
   identity across duplicate rows, geometry equality).
