"""
One-shot rebuild of hex_lookup_{res}.csv, snn2_{res}.json (+ feature summary
+ metrics), and hex_geom_{res}.geojson for all three resolutions, using the
corrected GRID_ID-based join in prep.py (see prep.py's module docstring).

Run from scripts/: python3 rebuild_pipeline.py
"""
import json
import os
import sys

import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from prep import prep_resolution
from train_snn2 import run_resolution
from snn2_inference import SNN2Model

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

RAW_GEOJSON = {
    "quarter": "/mnt/user-data/uploads/LA 311 Final data/documented_content/03_geo_feature_enrichment/enriched_outputs/Quarter Variables and Tents.geojson",
    "half": "/mnt/user-data/uploads/LA 311 Final data/documented_content/03_geo_feature_enrichment/enriched_outputs/Half Variables and Tents.geojson",
    "three_fourth": "/mnt/user-data/uploads/LA 311 Final data/documented_content/03_geo_feature_enrichment/enriched_outputs/Three Fourth Variables and Tent.geojson",
}
ENRICHED_CSV = {
    "quarter": os.path.join(DATA_DIR, "quarter_variables_and_tents_geo_enriched.csv"),
    "half": os.path.join(DATA_DIR, "half_variables_and_tents_geo_enriched.csv"),
    "three_fourth": os.path.join(DATA_DIR, "three_fourth_variables_and_tent_geo_enriched.csv"),
}
OUTPUTLAYER_CSV = {
    "quarter": os.path.join(DATA_DIR, "outputLayer_0_7.csv"),
    "half": os.path.join(DATA_DIR, "outputLayer_0_9.csv"),
    "three_fourth": os.path.join(DATA_DIR, "outputLayer_0_10.csv"),
}

results = []
for res in ["quarter", "half", "three_fourth"]:
    print(f"\n{'#'*70}\n# {res}\n{'#'*70}")

    # 1. Prep with corrected GRID_ID join
    hex_df = prep_resolution(ENRICHED_CSV[res], OUTPUTLAYER_CSV[res])
    print(f"  prepped {len(hex_df)} hexes, tent_present rate {hex_df['tent_present'].mean():.4f}")

    lookup_path = os.path.join(DATA_DIR, f"hex_lookup_{res}.csv")
    hex_df.to_csv(lookup_path, index=False)

    # 2. Train (overwrites models/snn2_{res}.json + summary + metrics)
    metrics = run_resolution(res, lookup_path, out_dir=MODEL_DIR)
    results.append(metrics)

    # 3. Score every hex with the freshly trained model -> baseline_risk +
    #    top_interaction_a/b/val, append to the same CSV.
    model = SNN2Model(os.path.join(MODEL_DIR, f"snn2_{res}.json"))
    hex_df["baseline_risk"] = model.predict_batch(hex_df)
    pairs, vals = model.batch_top_interactions(hex_df)
    hex_df["top_interaction_a"] = [p[0] if p else None for p in pairs]
    hex_df["top_interaction_b"] = [p[1] if p else None for p in pairs]
    hex_df["top_interaction_val"] = vals
    hex_df.to_csv(lookup_path, index=False)
    print(f"  wrote {lookup_path} ({len(hex_df.columns)} cols)")

    # 4. Real geometry, GRID_ID-keyed (no more "no data" filler -- every
    #    physical hex in the raw export now has a matching hex_lookup row).
    raw = gpd.read_file(RAW_GEOJSON[res])
    raw_dedup = raw.sort_values("OBJECTID").drop_duplicates(subset="GRID_ID", keep="first")
    geom_by_gid = raw_dedup.set_index("GRID_ID").geometry

    features = []
    n_missing_geom = 0
    for _, row in hex_df.iterrows():
        gid = row["GRID_ID"]
        if gid not in geom_by_gid.index:
            n_missing_geom += 1
            continue
        geom = geom_by_gid.loc[gid]
        features.append({
            "type": "Feature",
            "properties": {
                "OBJECTID": int(row["OBJECTID"]),
                "GRID_ID": gid,
                "has_data": True,
            },
            "geometry": json.loads(gpd.GeoSeries([geom]).to_json())["features"][0]["geometry"],
        })
    geom_fc = {"type": "FeatureCollection", "features": features}
    geom_path = os.path.join(DATA_DIR, f"hex_geom_{res}.geojson")
    with open(geom_path, "w") as fh:
        json.dump(geom_fc, fh)
    print(f"  wrote {geom_path} ({len(features)} hexes, {n_missing_geom} missing geometry)")

print("\n\nSUMMARY")
print(pd.DataFrame(results).to_string(index=False))
