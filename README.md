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
(cells 83/105/129) for architectural fidelity:

- **Feature lists restored verbatim**, including the 0.25mi resolution's
  12 extra "raw Esri alias" columns (11 are exact duplicates of data
  already present under a renamed column; one — veteran % — is a
  genuinely distinct column that was available but not previously
  included).
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

### Known remaining gap

Our retrained AUCs (quarter 0.786, half 0.811, three_fourth 0.848) still
don't exactly match the reference run (0.776 / 0.714 / 0.723) or the
paper's Table 2. Root cause identified: the reference notebook's
in-memory `y_binary` has tent-present rates (25.2% / 38.6% / 46.4%) that
exactly match the paper's Table 2 — ours (32.9% / 47.2% / 54.7%) are
higher. The underlying detection points are confirmed identical (1906 of
1908 verified points cross-match by filename), so the gap is in an
upstream point-to-hex spatial join step from earlier notebook cells
(1-82) that build `X_final`/`y_binary`, which weren't available to
reconstruct exactly. Everything downstream of that label (feature lists,
interaction architecture, training procedure) has been matched as
closely as possible.

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
2. See "Known remaining gap" above re: exact AUC reproduction.
