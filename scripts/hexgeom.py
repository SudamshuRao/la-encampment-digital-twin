"""
Builds an approximate regular-hexagon polygon (lat/lon ring) around each
hex's reconstructed centroid, sized from the known hex_area_sqkm so the
polygon is roughly the correct footprint even though we don't have the
original ArcGIS tessellation geometry.
"""
import numpy as np

KM_PER_DEG_LAT = 111.32


def hex_vertices(center_lat: float, center_lon: float, area_sqkm: float, n=6):
    """Pointy-top regular hexagon ring (closed, 7 points) in [lon, lat] pairs."""
    # area = (3*sqrt(3)/2) * R^2  =>  R = sqrt(area / (3*sqrt(3)/2))
    R_km = np.sqrt(area_sqkm / (3 * np.sqrt(3) / 2))
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(center_lat))

    angles = np.deg2rad(np.arange(0, 360, 60) + 30)  # pointy-top
    lats = center_lat + (R_km * np.sin(angles)) / KM_PER_DEG_LAT
    lons = center_lon + (R_km * np.cos(angles)) / km_per_deg_lon
    ring = list(zip(lons.tolist(), lats.tolist()))
    ring.append(ring[0])
    return ring


def hexes_to_geojson(df, risk_col="baseline_risk", id_col="GRID_ID"):
    """df needs: GRID_ID, centroid_lat, centroid_lon, hex_area_sqkm, risk_col.
    Optionally top_interaction_a/b (+ label columns) for the map tooltip."""
    from labels2 import FEATURE_LABELS_L2
    features = []
    has_interaction = "top_interaction_a" in df.columns
    for _, row in df.iterrows():
        ring = hex_vertices(row["centroid_lat"], row["centroid_lon"], row["hex_area_sqkm"])
        props = {
            "GRID_ID": row[id_col],
            "risk": round(float(row[risk_col]), 4),
            "tent_present": int(row.get("tent_present", 0)),
        }
        if has_interaction and row.get("top_interaction_a"):
            a, b = row["top_interaction_a"], row["top_interaction_b"]
            props["top_interaction"] = f"{FEATURE_LABELS_L2.get(a, a)} \u00d7 {FEATURE_LABELS_L2.get(b, b)}"
        else:
            props["top_interaction"] = "—"
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": features}
