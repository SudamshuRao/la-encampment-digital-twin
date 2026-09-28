"""
Shared prep utilities for the LA Encampment Risk Digital Twin.

Handles three things the raw exports don't give us for free:
  1. Merging each resolution's enriched export with its authoritative
     outputLayer_0_* join table on GRID_ID (see "GRID_ID vs OBJECTID" below)
     -- the enriched export's own Join_Count column is from a stale/
     different spatial join and does NOT match the paper's reported
     tent-present rates; the outputLayer merge below reproduces the
     reference notebook's 25.21/38.6/46.4% exactly.
  2. Renaming raw Esri/ACS columns to the names the notebook's FEATURES lists expect.
  3. Reconstructing an approximate centroid for every hex (including zero-detection
     ones) from the GRID_ID column-letter/row-number address, via affine regression.

GRID_ID vs OBJECTID (corrected join key, superseding an earlier version of
this pipeline)
-----------------------------------------------------------------------
This pipeline previously joined the enriched export to the outputLayer
table ON OBJECTID, on the assumption that OBJECTID was the reliable
cross-file key and GRID_ID was not. Both halves of that assumption turned
out to be backwards, confirmed by direct geometry-equality and
sjoin_nearest spatial tests against fresh raw exports:
  - OBJECTID is each file's own independent row-counter and has near-zero
    (~0.05-0.2%) correspondence to the same physical hex across two
    different ArcGIS export files. Joining on it silently pairs each
    hex's label with another, unrelated hex's features -- the label
    RATES it produces still looked right (they come straight from the
    outputLayer side untouched) but the FEATURES attached to each label
    were essentially random.
  - GRID_ID (e.g. "AS-43") *is* the stable, ~100%-matching spatial
    address across files, once deduplicated within a single file: the
    enriched export carries 2-4 rows per physical hex, one per street-
    view-image heading captured at that hex (confirmed: every
    demographic/crime/facility/hazard/climate feature column is
    byte-identical across a GRID_ID's duplicate rows; only per-photo
    metadata -- filename, heading, lat/lon, address, source_run, and
    sometimes tent_prediction_count -- differs), so keeping one row per
    GRID_ID loses no feature information.
  - The outputLayer table's GRID_ID set matches the deduplicated enriched
    export's GRID_ID set exactly (1448/1448, 767/767, 539/539 for
    quarter/half/three_fourth) -- outputLayer already covers every
    physical hex in the study grid, including Join_Count == 0 hexes, not
    just the ones with detections.
"""
import re
import numpy as np
import pandas as pd
from numpy.linalg import lstsq

RENAME_MAP = {
    "CRMCYBURG": "burglary_crime_est",
    "CRMCYTOTC": "total_crime_est",
    "TOTPOP_CY": "total_population",
    "UNEMPRT_CY_I": "unemployment_rate_idx",
    "HISPPOP_CY_P": "pct_hispanic",
    "MEDHINC_CY_I": "median_income_idx",
    "ACSVET_P": "pct_veterans",
    "ACSVET_P_1": "pct_veterans_2",
    "ACSHSGRAD_P": "pct_hs_grad",
    "ACS35NOHI_P": "pct_no_health_insurance",
    "ACSEDNOSCH_P": "pct_no_schooling",
    "ACSGRNTI50_P": "pct_severe_rent_burden",
    "ACSBLWPV_P": "pct_below_poverty",
    "ACSPUBAI_P": "pct_public_assistance",
    "ACSHHDIS_P": "pct_household_disability",
    "X14068_X_I": "esri_market_idx_14068",
    "X14068FY_X_I": "esri_market_idx_14068_forecast",
}

GRID_RE = re.compile(r"^([A-Z]+)-(\d+)$")


def colletter_to_num(s: str) -> int:
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def parse_grid_id(gid: str):
    m = GRID_RE.match(gid)
    if not m:
        return None, None
    return colletter_to_num(m.group(1)), int(m.group(2))


def load_and_merge_hexes(enriched_csv_path: str, outputlayer_csv_path: str) -> pd.DataFrame:
    """Merge a resolution's enriched export with its authoritative outputLayer_0_*
    join table on GRID_ID (see module docstring for why this replaced an
    OBJECTID-based join).

    The enriched export carries 2-4 rows per physical hex (one per street-
    view heading); every non-photo feature column is identical across a
    GRID_ID's duplicate rows, so we deduplicate it down to one row per
    GRID_ID (keeping the first, arbitrarily -- it doesn't matter which,
    since they're identical) before joining. The outputLayer table is
    already exactly one row per GRID_ID on its own side.

    Join_Count / TARGET_FID collide between the two files and get pandas's
    suffixes:
      - Join_Count: keep the outputLayer side -- it's the authoritative
        label source and matches the paper's tent-present rates exactly.
      - OBJECTID: keep the enriched-export side -- it's paired with that
        same row's own latitude/longitude, which centroid reconstruction
        below needs to stay internally consistent, and it's the column the
        rest of the app (hexgeom.py, scenarios.py) uses as each hex's
        unique in-file key.
    Everything else suffixed is just duplicate metadata from the two Esri
    enrichment passes and is dropped.
    """
    df_tent = pd.read_csv(outputlayer_csv_path)
    df_raw = pd.read_csv(enriched_csv_path)
    df_raw = df_raw.sort_values("OBJECTID").drop_duplicates(subset="GRID_ID", keep="first")

    merged = pd.merge(df_tent, df_raw, on="GRID_ID", how="inner", suffixes=("_tent", "_raw"))

    merged = merged.assign(Join_Count=merged["Join_Count_tent"], OBJECTID=merged["OBJECTID_raw"])
    drop_suffixed = [c for c in merged.columns
                      if c.endswith(("_tent", "_raw")) and c not in ("Join_Count", "OBJECTID")]
    merged = merged.drop(columns=drop_suffixed)
    merged = merged.dropna(subset=["GRID_ID"]).copy()

    colnum, rownum = zip(*merged["GRID_ID"].map(parse_grid_id))
    merged = merged.assign(colnum=colnum, row=rownum)
    merged = merged.rename(columns=RENAME_MAP)
    merged = merged.copy()  # defragment after the renames/assigns above
    merged["school_count"] = merged["public_school_count"].fillna(0) + merged["private_school_count"].fillna(0)
    merged["tent_present"] = (merged["Join_Count"] > 0).astype(int)
    return merged


def reconstruct_centroids(hex_df: pd.DataFrame) -> pd.DataFrame:
    """Fit (colnum,row)->(lat,lon) on hexes with known detection-averaged
    coordinates, then predict a centroid for every hex, including hexes with
    zero detections (which have no coordinates at all in the raw export)."""
    known = hex_df.dropna(subset=["latitude", "longitude"])
    A = np.column_stack([known["colnum"], known["row"], np.ones(len(known))])
    coef_lat = lstsq(A, known["latitude"], rcond=None)[0]
    coef_lon = lstsq(A, known["longitude"], rcond=None)[0]

    A_full = np.column_stack([hex_df["colnum"], hex_df["row"], np.ones(len(hex_df))])
    hex_df = hex_df.copy()
    hex_df["centroid_lat"] = A_full @ coef_lat
    hex_df["centroid_lon"] = A_full @ coef_lon
    hex_df["centroid_source"] = np.where(
        hex_df["latitude"].notna(), "detection_avg", "reconstructed"
    )
    return hex_df


def prep_resolution(enriched_csv_path: str, outputlayer_csv_path: str) -> pd.DataFrame:
    hex_df = load_and_merge_hexes(enriched_csv_path, outputlayer_csv_path)
    hex_df = reconstruct_centroids(hex_df)
    return hex_df
