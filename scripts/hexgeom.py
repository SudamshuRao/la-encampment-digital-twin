"""
Real ArcGIS hex geometry for the Streamlit map.

Each resolution's `hex_geom_{res}.geojson` (precomputed from the actual
ArcGIS-exported hex layer, not a reconstructed approximation) covers the
FULL tessellation ArcGIS produced -- roughly twice as many hexes as
`hex_lookup_{res}.csv`, because only ~half of the raw hexes had a matching
row in both the enriched export and the outputLayer join table (see
prep.py / README "Resolved gaps"). The other half genuinely have no model
data, not a rendering bug -- they're kept in the geometry file (flagged
`has_data: false`) purely so the map reads as one contiguous grid instead
of a patchwork of disconnected islands, matching how the reference deck's
own 0.25mi maps look. They're rendered as flat "no data" hexes with no
risk color, since there is no risk value to show.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

_GEOM_CACHE = {}


def load_geom(res):
    """Load (and cache in-process) the full real-geometry FeatureCollection
    for a resolution: every ArcGIS hex, has_data flag, no risk baked in."""
    if res not in _GEOM_CACHE:
        with open(os.path.join(DATA_DIR, f"hex_geom_{res}.geojson")) as f:
            _GEOM_CACHE[res] = json.load(f)
    return _GEOM_CACHE[res]


def hexes_to_geojson_real(geom_fc, display_df, risk_col="baseline_risk", id_col="OBJECTID"):
    """Merge live model data (display_df, one row per hex WITH data -- may
    include a slider-driven risk override for the selected hex) onto the
    real ArcGIS geometry. Hexes with no model data get has_data=False and a
    null risk; the caller's style_function is expected to render those as
    plain "no data" hexes rather than coloring them by risk.

    Joined on OBJECTID, not GRID_ID: GRID_ID is NOT a unique key in the
    enriched export (~60% of hexes share their GRID_ID address with at
    least one other, physically different hex -- e.g. quarter-resolution
    OBJECTID 2 and 3 are both "AQ-81"). OBJECTID is the only column
    guaranteed one-to-one with a real hex, in both the geometry file and
    hex_lookup_{res}.csv, so it's the only safe join key here. GRID_ID is
    still carried through purely for display (tooltip / dropdown), which
    is a pre-existing ambiguity elsewhere in this app worth being aware
    of, but out of scope for this geometry fix.
    """
    from labels2 import FEATURE_LABELS_L2
    lookup = display_df.set_index(id_col)
    has_interaction = "top_interaction_a" in display_df.columns

    features = []
    for feat in geom_fc["features"]:
        props_in = feat["properties"]
        oid = props_in["OBJECTID"]
        gid = props_in.get("GRID_ID")
        has_data = bool(props_in.get("has_data")) and oid in lookup.index

        if has_data:
            row = lookup.loc[oid]
            risk = round(float(row[risk_col]), 4)
            tent_present = int(row.get("tent_present", 0))
            if has_interaction and row.get("top_interaction_a"):
                a, b = row["top_interaction_a"], row["top_interaction_b"]
                top_interaction = f"{FEATURE_LABELS_L2.get(a, a)} × {FEATURE_LABELS_L2.get(b, b)}"
            else:
                top_interaction = "—"
            risk_display = f"{risk:.1%}"
        else:
            risk = None
            tent_present = 0
            top_interaction = "No data"
            risk_display = "No data"

        features.append({
            "type": "Feature",
            "properties": {
                "GRID_ID": gid if gid is not None else f"OBJ-{props_in['OBJECTID']}",
                "risk": risk,
                "risk_display": risk_display,
                "tent_present": tent_present,
                "top_interaction": top_interaction,
                "has_data": has_data,
            },
            "geometry": feat["geometry"],
        })
    return {"type": "FeatureCollection", "features": features}
