"""
SNN-2 training, matching the reference notebook pipeline (cells 83/105/129)
as closely as possible in pure NumPy:
  - FEATURES lists restored verbatim from the notebook, including the 12
    extra "raw Esri alias" columns in the 0.25mi resolution (11 are exact
    duplicates of data we already have under a renamed column; one --
    veteran % -- is a genuinely different column we have but hadn't
    included before).
  - SNN-1 main-effect layer trained first (rounds=4, matching reference),
    then FROZEN -- SNN-2 only fits interaction subnets on the residual,
    it does not retrain the main effects in the interaction layer's
    presence (this differs from an earlier version of this pipeline that
    trained base+interactions jointly).
  - Interactions use a true 2D joint-RBF kernel (BivariateRBFSubNet, 6x6
    basis grid) over the actual pair of raw feature values, not a
    collapsed single "product" column.
  - Pair selection ranks by fitting a shallow RF directly on the product
    columns against hard labels (not soft labels, not combined with base
    features) -- matches select_top_pairs in the reference exactly.
  - Plain random StratifiedKFold CV, not spatially blocked -- the
    reference code doesn't spatially block, and matches the paper's
    published numbers closely without it.
"""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from snn_core import train_additive_snn, train_interaction_snn, select_top_pairs

warnings.filterwarnings("ignore")
np.random.seed(42)

# Aliases needed only for the 0.25mi resolution: 11 of these 12 raw-Esri-named
# columns are exact duplicates of a column we already have under its renamed
# name; "2023 Civilian Pop 18+: Veteran..." is the genuinely distinct
# first-Esri-pass veteran% (pct_veterans, as opposed to pct_veterans_2).
QUARTER_ALIASES = {
    "2024 Total Crime Index": "total_crime_est",
    "2024 Total Population": "total_population",
    "2024 Unemployment Rate: Index": "unemployment_rate_idx",
    "2023 HHs w/1+ Persons w/Disability (ACS 5-Yr): Percent": "pct_household_disability",
    "2023 HHs w/Public Assist Income (ACS 5-Yr): Percent": "pct_public_assistance",
    "2023 Pop w/Income Below Poverty Level (ACS 5-Yr): Percent": "pct_below_poverty",
    "2023 HHs/Gross Rent 50+% of Income (ACS 5-Yr): Percent": "pct_severe_rent_burden",
    "2023 Pop 25+: No Schooling (ACS 5-Yr): Percent": "pct_no_schooling",
    "2023 Pop 35-64: No Health Insurance (ACS 5-Yr): Percent": "pct_no_health_insurance",
    "2023 Civilian Pop 18+: Veteran (ACS 5-Yr): Percent": "pct_veterans",
    "2023 Pop 25+: HS Diploma (ACS 5-Yr): Percent": "pct_hs_grad",
    "2030 Value of Credit Card Debt: Index": "esri_market_idx_14068_forecast",
}

FEATURES_BY_RES = {
    "quarter": [
        "2024 Total Crime Index", "2024 Total Population", "2024 Unemployment Rate: Index",
        "2023 HHs w/1+ Persons w/Disability (ACS 5-Yr): Percent",
        "2023 HHs w/Public Assist Income (ACS 5-Yr): Percent",
        "2023 Pop w/Income Below Poverty Level (ACS 5-Yr): Percent",
        "2023 HHs/Gross Rent 50+% of Income (ACS 5-Yr): Percent",
        "2023 Pop 25+: No Schooling (ACS 5-Yr): Percent",
        "2023 Pop 35-64: No Health Insurance (ACS 5-Yr): Percent",
        "2023 Civilian Pop 18+: Veteran (ACS 5-Yr): Percent",
        "2023 Pop 25+: HS Diploma (ACS 5-Yr): Percent",
        "2030 Value of Credit Card Debt: Index",
        "total_crime_est", "total_population", "burglary_crime_est", "unemployment_rate_idx",
        "pct_household_disability", "pct_public_assistance", "pct_below_poverty",
        "pct_severe_rent_burden", "pct_no_schooling", "pct_no_health_insurance",
        "pct_veterans_2", "pct_hs_grad", "esri_market_idx_14068_forecast",
        "hospital_count", "fire_station_count", "library_count", "college_count",
        "public_school_count", "private_school_count", "recreation_center_count",
        "adult_recreation_count", "affordable_housing_count", "sud_treatment_count",
        "police_station_count", "food_access_count", "lausd_school_site_count",
        "nearest_osm_major_road_m", "freeway_segments_count", "nearest_freeway_m",
        "river_drainage_segments_count", "nearest_river_drainage_m", "osm_amenity_count",
        "fema_flood_hazard_area_pct", "la_geohub_100yr_flood_intersects",
        "la_geohub_500yr_flood_area_pct", "la_geohub_500yr_flood_intersects",
        "la_geohub_fire_hazard_area_pct", "tree_canopy_proxy_pct", "water_proxy_pct",
        "bare_ground_proxy_pct", "dominant_landcover_class",
    ],
    "half": [
        "burglary_crime_est", "total_crime_est", "total_population", "unemployment_rate_idx",
        "pct_veterans", "esri_market_idx_14068_forecast", "pct_hs_grad",
        "pct_no_health_insurance", "pct_no_schooling", "pct_severe_rent_burden",
        "pct_below_poverty", "pct_public_assistance", "pct_household_disability",
        "transit_stops_count", "hospital_count", "fire_station_count", "library_count",
        "college_count", "public_school_count", "private_school_count",
        "recreation_center_count", "adult_recreation_count", "affordable_housing_count",
        "sud_treatment_count", "business_sites_count", "police_station_count",
        "food_access_count", "nearest_osm_major_road_m", "freeway_segments_count",
        "nearest_freeway_m", "river_drainage_segments_count", "osm_amenity_count",
        "fema_flood_hazard_area_pct", "la_geohub_100yr_flood_area_pct",
        "la_geohub_100yr_flood_intersects", "la_geohub_500yr_flood_area_pct",
        "la_geohub_500yr_flood_intersects", "la_geohub_fire_hazard_area_pct",
        "tree_canopy_proxy_pct", "water_proxy_pct", "bare_ground_proxy_pct",
        "dominant_landcover_class",
    ],
    "three_fourth": [
        "total_crime_est", "total_population", "unemployment_rate_idx",
        "pct_household_disability", "pct_public_assistance", "pct_severe_rent_burden",
        "pct_no_schooling", "pct_no_health_insurance", "pct_veterans_2", "pct_hs_grad",
        "ACSHHPBLPV_P", "esri_market_idx_14068_forecast", "hospital_count",
        "fire_station_count", "library_count", "college_count", "public_school_count",
        "private_school_count", "recreation_center_count", "adult_recreation_count",
        "affordable_housing_count", "sud_treatment_count", "business_sites_count",
        "police_station_count", "food_access_count", "lausd_school_site_count",
        "nearest_osm_major_road_m", "freeway_segments_count", "nearest_freeway_m",
        "river_drainage_segments_count", "nearest_river_drainage_m", "osm_amenity_count",
        "fema_flood_hazard_area_pct", "fema_flood_hazard_intersects",
        "la_geohub_100yr_flood_area_pct", "la_geohub_100yr_flood_intersects",
        "la_geohub_500yr_flood_area_pct", "la_geohub_500yr_flood_intersects",
        "la_geohub_fire_hazard_area_pct", "tree_canopy_proxy_pct", "water_proxy_pct",
        "bare_ground_proxy_pct", "dominant_landcover_class",
    ],
}

TOP_K_INTERACTIONS = 20   # matches reference default (args.top_pairs)
N_RBF_MAIN = 10
N_ROUNDS_MAIN = 4         # matches reference (args.rounds)
N_RBF2 = 6                # matches reference (args.rbf2)
N_ROUNDS2 = 3              # matches reference (args.rounds2)


def run_resolution(res_name: str, hex_csv: str, out_dir: str = "models",
                    top_k: int = TOP_K_INTERACTIONS):
    print(f"\n{'='*60}\n{res_name.upper()} -- SNN-2\n{'='*60}")
    df = pd.read_csv(hex_csv)

    if res_name == "quarter":
        for alias, source in QUARTER_ALIASES.items():
            df[alias] = df[source]

    base_features = FEATURES_BY_RES[res_name]
    missing = [f for f in base_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X_base = df[base_features].fillna(0).astype(float)
    y = df["tent_present"].values
    print(f"  {len(X_base)} hexes | {len(base_features)} base features | "
          f"{y.sum()} tent-present ({y.mean()*100:.1f}%)")

    scaler_base = StandardScaler()
    X_base_scaled = scaler_base.fit_transform(X_base)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_probs = cross_val_predict(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
                                  X_base_scaled, y, cv=cv, method="predict_proba")[:, 1]
    print(f"  LogReg AUC (CV):  {roc_auc_score(y, lr_probs):.4f}")

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=8,
                                 random_state=42, n_jobs=-1)
    soft_labels = cross_val_predict(rf, X_base_scaled, y, cv=cv, method="predict_proba")[:, 1]
    rf.fit(X_base_scaled, y)
    rf_auc = roc_auc_score(y, soft_labels)
    print(f"  RF teacher AUC (CV): {rf_auc:.4f}")

    print(f"  training SNN-1 (rbf={N_RBF_MAIN}, rounds={N_ROUNDS_MAIN})...")
    nets1, contribs1, probs1 = train_additive_snn(X_base_scaled, soft_labels, y,
                                                    n_rbf=N_RBF_MAIN, n_rounds=N_ROUNDS_MAIN)
    auc1 = roc_auc_score(y, probs1)
    print(f"  SNN-1 AUC: {auc1:.4f}")

    print(f"  selecting top {top_k} pairwise interactions...")
    top_pairs = select_top_pairs(X_base_scaled, y, len(base_features), top_k=top_k)
    j0, k0 = top_pairs[0]
    print(f"  top pair: {base_features[j0]} x {base_features[k0]}")

    print(f"  training interaction layer (rbf2={N_RBF2}, rounds={N_ROUNDS2})...")
    pair_nets, pair_contribs, probs2 = train_interaction_snn(
        X_base_scaled, soft_labels, y, contribs1.sum(1), top_pairs,
        n_rbf2=N_RBF2, n_rounds=N_ROUNDS2,
    )

    auc2 = roc_auc_score(y, probs2)
    ap2 = average_precision_score(y, probs2)
    fpr, tpr, thresholds = roc_curve(y, probs2)
    thresh2 = float(thresholds[np.argmax(tpr - fpr)])
    report2 = classification_report(y, (probs2 >= thresh2).astype(int), target_names=["no tent", "tent"])
    print(f"  SNN-2 AUC: {auc2:.4f}  AP: {ap2:.4f}  (lift {auc1:.4f} -> {auc2:.4f})")

    delta1 = contribs1[y == 1].mean(0) - contribs1[y == 0].mean(0)
    mean_abs1 = np.abs(contribs1).mean(0)
    main_summary = pd.DataFrame({
        "feature": base_features, "is_interaction": False,
        "mean_abs_sj": mean_abs1, "delta_sj": delta1,
        "raw_min": X_base.min().values, "raw_max": X_base.max().values,
        "raw_p05": X_base.quantile(0.05).values, "raw_p95": X_base.quantile(0.95).values,
    })

    inter_rows = []
    for idx, (j, k) in enumerate(top_pairs):
        col = pair_contribs[:, idx]
        name = f"{base_features[j]}__x__{base_features[k]}"
        inter_rows.append({
            "feature": name, "is_interaction": True,
            "mean_abs_sj": np.abs(col).mean(),
            "delta_sj": col[y == 1].mean() - col[y == 0].mean(),
            "raw_min": np.nan, "raw_max": np.nan, "raw_p05": np.nan, "raw_p95": np.nan,
        })
    inter_summary = pd.DataFrame(inter_rows)
    summary = pd.concat([main_summary, inter_summary], ignore_index=True).sort_values(
        "delta_sj", ascending=False).reset_index(drop=True)

    model_artifact = {
        "base_features": base_features,
        "interaction_pairs": [[base_features[j], base_features[k]] for j, k in top_pairs],
        "scaler_base_mean": scaler_base.mean_.tolist(),
        "scaler_base_scale": scaler_base.scale_.tolist(),
        "main_nets": [n.state_dict() for n in nets1],
        "pair_nets": [pair_nets[jk].state_dict() for jk in top_pairs],
        "threshold": thresh2,
        "n_rbf": N_RBF_MAIN,
        "n_rbf2": N_RBF2,
    }
    with open(f"{out_dir}/snn2_{res_name}.json", "w") as fh:
        json.dump(model_artifact, fh)
    summary.to_csv(f"{out_dir}/snn2_feature_summary_{res_name}.csv", index=False)
    with open(f"{out_dir}/snn2_metrics_{res_name}.txt", "w") as fh:
        fh.write(f"LogReg AUC (CV):     {roc_auc_score(y, lr_probs):.4f}\n")
        fh.write(f"RF teacher AUC (CV): {rf_auc:.4f}\n")
        fh.write(f"SNN-1 AUC:           {auc1:.4f}\n")
        fh.write(f"SNN-2 AUC:           {auc2:.4f}\n")
        fh.write(f"SNN-2 AP:            {ap2:.4f}\n")
        fh.write(f"Optimal threshold:   {thresh2:.3f}\n\n")
        fh.write(report2)

    return {"res": res_name, "n_hexes": len(X_base), "n_base_features": len(base_features),
            "n_interactions": len(top_pairs), "lr_auc": round(roc_auc_score(y, lr_probs), 4),
            "rf_auc": round(rf_auc, 4), "snn1_auc": round(auc1, 4), "snn2_auc": round(auc2, 4)}


if __name__ == "__main__":
    results = []
    for res in ["quarter", "half", "three_fourth"]:
        results.append(run_resolution(res, f"data/hex_lookup_{res}.csv"))
    print("\n\nSUMMARY")
    print(pd.DataFrame(results).to_string(index=False))
