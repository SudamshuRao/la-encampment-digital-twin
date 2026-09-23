"""
Population-level what-if scenario optimization, per the professor's
"Simple SNN-2 What-If Digital Twin" doc's variable-role table:
  - Frozen context (do not change): socioeconomic, demographic, crime,
    hazard, roadway, hydrology, land cover.
  - Scenario inputs (change only for what-if scenarios): the doc lists
    "affordable housing access, food access, selected service/amenity
    access" as EXAMPLES, not an exhaustive list -- this module uses the
    full non-frozen set (all Facilities/Services variables), matching
    the same candidate set the "Actionable" optimizer already uses.
  - Model output: predicted encampment probability, recalculated with SNN-2.

Applies the existing greedy optimizer (optimizer2.greedy_optimize_snn2)
independently to each of the highest-risk N% hexes, then aggregates the
results across that population: mean risk before/after, and which
variables got recommended most often.
"""
import numpy as np
import pandas as pd

from optimizer2 import greedy_optimize_snn2


def select_target_hexes(hex_df: pd.DataFrame, top_pct: float) -> pd.Series:
    """top_pct as a fraction (e.g. 0.10 for top 10%). Returns a boolean mask."""
    threshold = hex_df["baseline_risk"].quantile(1 - top_pct)
    return hex_df["baseline_risk"] >= threshold


def run_population_optimization(model, hex_df: pd.DataFrame, top_pct: float,
                                 candidate_features: list, bounds: dict, k: int = 3):
    """Runs the greedy Actionable-style optimizer independently on every
    targeted hex, then aggregates. Returns (results_df, summary_row,
    variable_usage_df)."""
    target_mask = select_target_hexes(hex_df, top_pct)
    targeted = hex_df[target_mask].copy()

    rows = []
    usage_counts = {f: 0 for f in candidate_features}
    for _, hex_row in targeted.iterrows():
        current = {f: (float(hex_row[f]) if pd.notna(hex_row[f]) else 0.0) for f in model.base_features}
        result = greedy_optimize_snn2(model, current, candidate_features, bounds, k=k, allow_decrease=False)
        for s in result["steps"]:
            usage_counts[s["feature"]] += 1
        rows.append({
            "GRID_ID": hex_row["GRID_ID"],
            "baseline_risk": result["baseline_risk"],
            "optimized_risk": result["optimized_risk"],
            "delta": result["baseline_risk"] - result["optimized_risk"],
            "n_steps_used": len(result["steps"]),
        })

    results_df = pd.DataFrame(rows)
    full = hex_df[["GRID_ID"]].copy()
    full = full.merge(results_df, on="GRID_ID", how="left")
    full["is_target"] = target_mask.values
    full["baseline_risk"] = full["baseline_risk"].fillna(hex_df["baseline_risk"])
    full["optimized_risk"] = full["optimized_risk"].fillna(full["baseline_risk"])
    full["delta"] = full["delta"].fillna(0.0)
    full["n_steps_used"] = full["n_steps_used"].fillna(0).astype(int)

    summary_row = {
        "n_hexes_targeted": int(target_mask.sum()),
        "mean_baseline_risk_targeted": float(results_df["baseline_risk"].mean()) if len(results_df) else 0.0,
        "mean_optimized_risk_targeted": float(results_df["optimized_risk"].mean()) if len(results_df) else 0.0,
        "mean_reduction_targeted": float(results_df["delta"].mean()) if len(results_df) else 0.0,
    }

    from labels2 import FEATURE_LABELS_L2
    usage_df = pd.DataFrame({
        "feature": list(usage_counts.keys()),
        "label": [FEATURE_LABELS_L2.get(f, f) for f in usage_counts],
        "times_recommended": list(usage_counts.values()),
    })
    n_targeted = max(int(target_mask.sum()), 1)
    usage_df["pct_of_targeted_hexes"] = usage_df["times_recommended"] / n_targeted
    usage_df = usage_df[usage_df["times_recommended"] > 0].sort_values("times_recommended", ascending=False)

    return full, summary_row, usage_df
