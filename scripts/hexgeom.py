"""
Real ArcGIS hex geometry for the Streamlit map.

Each resolution's `hex_geom_{res}.geojson` is built (by scripts/
rebuild_pipeline.py) from the real ArcGIS-exported hex polygons,
deduplicated to one polygon per GRID_ID and joined 1:1 against
`hex_lookup_{res}.csv`'s rows (both files share the same OBJECTID scheme
because the geometry file is written directly from the same prepped
dataframe). Every physical hex in the study grid has a `hex_lookup` row
now -- the corrected GRID_ID-based join in prep.py resolved what earlier
looked like ~half the hexes genuinely having no model data (see
prep.py's module docstring / README "Resolved gaps"): that was an
artifact of the old OBJECTID-based cross-file join, not real missing
coverage. `has_data` is kept in the schema for the map's style function
but is always true now; retained rather than removed so app.py's
rendering path doesn't need touching.
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

    Joined on OBJECTID: both files are written from the same prepped
    dataframe (see rebuild_pipeline.py), so OBJECTID and GRID_ID are both
    1:1 with a real hex here -- the enriched export's raw duplicate-
    GRID_ID rows (multiple street-view headings per physical hex) were
    already collapsed to one row per hex during prep, before either file
    was written.
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
                "OBJECTID": oid,
                "GRID_ID": gid if gid is not None else f"OBJ-{oid}",
                "risk": risk,
                "risk_display": risk_display,
                "tent_present": tent_present,
                "top_interaction": top_interaction,
                "has_data": has_data,
            },
            "geometry": feat["geometry"],
        })
    return {"type": "FeatureCollection", "features": features}
