"""
Shared prep utilities for the LA Encampment Risk Digital Twin.

Handles three things the raw exports don't give us for free:
  1. Collapsing multi-row-per-hex detection-point exports into one row per hex.
  2. Renaming raw Esri/ACS columns to the names the notebook's FEATURES lists expect.
  3. Reconstructing an approximate centroid for every hex (including zero-detection
     ones) from the GRID_ID column-letter/row-number address, via affine regression.
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


def load_and_dedup_hexes(csv_path: str) -> pd.DataFrame:
    """Collapse a raw detection-point-level export to one row per GRID_ID hex."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["GRID_ID"]).copy()

    colnum, rownum = zip(*df["GRID_ID"].map(parse_grid_id))
    df["colnum"] = colnum
    df["row"] = rownum

    id_cols = ["OBJECTID", "TARGET_FID", "JOIN_FID"]
    non_feature_cols = id_cols + ["latitude", "longitude", "address", "heading",
                                   "filename", "source_run", "tent_prediction_count",
                                   "max_conf", "classes_seen", "Join_Count"]

    feature_cols = [c for c in df.columns if c not in non_feature_cols
                     and c not in ["GRID_ID", "colnum", "row"]]

    agg = df.groupby("GRID_ID", as_index=False).agg({
        **{c: "first" for c in feature_cols},
        "colnum": "first",
        "row": "first",
        "Join_Count": "max",
        "latitude": "mean",
        "longitude": "mean",
    })
    agg = agg.rename(columns=RENAME_MAP)
    agg["school_count"] = agg["public_school_count"].fillna(0) + agg["private_school_count"].fillna(0)
    agg["tent_present"] = (agg["Join_Count"] > 0).astype(int)
    return agg


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


def prep_resolution(csv_path: str) -> pd.DataFrame:
    hex_df = load_and_dedup_hexes(csv_path)
    hex_df = reconstruct_centroids(hex_df)
    return hex_df
